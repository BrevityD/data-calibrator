import sys
import unittest
from evaluation.core import extract_code, check_syntax_and_entry_point, evaluate_code_correctness

# Test Extraction
text = """
Here is the code:
```python
def solve():
    return 42
```
"""
extracted = extract_code(text)
print(f"Extracted: {extracted.strip()}")

# Test Syntax
print(f"Syntax Check (solve): {check_syntax_and_entry_point(extracted, 'solve')}")
print(f"Syntax Check (foo): {check_syntax_and_entry_point(extracted, 'foo')}")

# Test Execution (Mocking example)
example = {
    "entry_point": "solve",
    "test": """
class Test(unittest.TestCase):
    def test_answer(self):
        self.assertEqual(solve(), 42)
""",
    "libs": ""
}

# Correct prediction
print("Eval Correct:", evaluate_code_correctness(text, example))

# Incorrect prediction
bad_text = """
```python
def solve():
    return 0
```
"""
print("Eval Incorrect:", evaluate_code_correctness(bad_text, example))

# Syntax Error
err_text = "def solve( return 0"
print("Eval Syntax Error:", evaluate_code_correctness(err_text, example))
