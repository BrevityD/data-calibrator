import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class GradientEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if self.model:
            self.model.eval()
            self.model.to(self.device)

    def evaluate(self, dataset, batch_size=4, max_samples=None, max_length=2048, description="Evaluating Gradients"):
        """
        Computes the gradient of the loss on the dataset w.r.t model parameters
        and returns statistics of this aggregated gradient.
        """
        self.model.eval()
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        def collate_fn(batch):
            input_ids_list = []
            labels_list = []
            
            for item in batch:
                # Construct conversation
                # The code_adaptor returns 'prompt' and 'completion' lists of messages
                messages = item['prompt'] + item['completion']
                
                try:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except Exception:
                    # Fallback
                    text = ""
                    for msg in messages:
                        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                
                encodings = self.tokenizer(
                    text, 
                    max_length=max_length, 
                    truncation=True, 
                    padding=False, 
                    return_tensors="pt"
                )
                
                input_ids = encodings.input_ids[0]
                labels = input_ids.clone()
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)
            
            # Pad
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, 
                batch_first=True, 
                padding_value=self.tokenizer.pad_token_id
            )
            labels_padded = torch.nn.utils.rnn.pad_sequence(
                labels_list, 
                batch_first=True, 
                padding_value=-100
            )
            
            attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)
            
            return {
                "input_ids": input_ids_padded,
                "labels": labels_padded,
                "attention_mask": attention_mask
            }

        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn, 
            shuffle=False,
            drop_last=False
        )
        
        # Zero gradients before accumulation
        self.model.zero_grad()
        
        total_loss_sum = 0.0
        total_tokens = 0
        
        for batch in tqdm(dataloader, desc=description):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Calculate number of valid tokens in the batch (considering shifting)
            # Models usually shift labels by 1, so the last token doesn't contribute, 
            # and the first label corresponds to prediction from first input.
            # effectively labels[:, 1:] are the targets.
            if batch["labels"] is not None:
                # Count non-pad tokens in the shifted labels
                shift_labels = batch["labels"][..., 1:].contiguous()
                num_tokens = (shift_labels != -100).sum().item()
            else:
                # Fallback if no labels provided (though loss wouldn't be computed anyway)
                num_tokens = 1

            if num_tokens == 0:
                continue

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # loss is the mean loss per token. 
            # We want to accumulate the sum of gradients, then divide by total tokens at the end.
            # So we multiply by num_tokens to get the "sum" loss for this batch.
            scaled_loss = loss * num_tokens
            
            scaled_loss.backward()
            
            total_loss_sum += scaled_loss.item()
            total_tokens += num_tokens
            
        avg_loss = total_loss_sum / total_tokens if total_tokens > 0 else 0.0
        
        # Collect gradients
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                # Normalize to get gradient of the mean loss (per token)
                if total_tokens > 0:
                    param.grad.div_(total_tokens)
                
                # Flatten and move to CPU to save GPU memory
                all_grads.append(param.grad.view(-1).cpu())
                
                # Clear grad
                param.grad = None
        
        if not all_grads:
            return {}

        flat_grads = torch.cat(all_grads).float() # Use float32 for stats
        
        # Compute Statistics
        metrics = {
            "loss": avg_loss,
            "total_tokens": total_tokens,
            "grad_norm": torch.norm(flat_grads, p=2).item(),
            "grad_mean": torch.mean(flat_grads).item(),
            "grad_var": torch.var(flat_grads).item(),
            "grad_max": torch.max(flat_grads).item(),
            "grad_min": torch.min(flat_grads).item(),
            "grad_l1_norm": torch.norm(flat_grads, p=1).item(),
            "grad_num_params": flat_grads.numel()
        }
        
        return metrics
