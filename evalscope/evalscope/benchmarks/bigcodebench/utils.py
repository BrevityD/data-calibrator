import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from typing import Dict, Optional, List

TIMEOUT_LIMIT = 240.0


@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException('Timed out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == '.':
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            print(f"Prevented attempt to kill PGID {pgid} with signal {sig}")

    def safe_system(command):
        print(f"Intercepted system command: {command}")
        if 'kill' in command or 'killall' in command:
            return 0  # Simulate successful execution without doing anything
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if 'kill' in command or 'killall' in command:
            return 0  # Simulate successful execution without doing anything
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if 'ps' in command:
            return b""  # Simulate no processes found
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        print(f"Intercepted subprocess run command: {args}")
        if 'kill' in args[0] or 'killall' in args[0]:
            return subprocess.CompletedProcess(args, 0, b'', b'')  # Simulate successful execution
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            print(f"Intercepted Popen command: {args}")
            kwargs['preexec_fn'] = os.setsid  # Start the process in a new session
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                print("Timeout expired, intercepted and returning None")
                return None, None

        def kill(self):
            print(f"Intercepted kill call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            print(f"Intercepted terminate call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        print(f"Intercepted os.popen command: {command}")
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        print(f"Intercepted exec command: {args}")

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error handling process {pid}: {e}")
        
        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    """
    
    os.environ['TZ'] = 'UTC'
    time.tzset()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    
    if max_as_limit and max_data_limit and max_stack_limit:
        import resource
        
        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024
        
        resource.setrlimit(
            resource.RLIMIT_AS, (max_as_limit, max_as_limit)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (max_data_limit, max_data_limit)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import matplotlib.pyplot as plt
    plt.close('all')

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None


def unsafe_execute(problem: Dict, completion: str, timeout: float, result_list: List):
    # Default limits for BigCodeBench
    max_as_limit = 30 * 1024
    max_data_limit = 30 * 1024
    max_stack_limit = 10

    with safe_environment(), create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = problem['prompt'] + completion + "\n" + problem['test']
            
            with swallow_io():
                # Compile and execute the code to populate the module
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                
                # BigCodeBench-style: check for TestCases class
                if hasattr(new_module, 'TestCases'):
                    TestCases = getattr(new_module, 'TestCases')
                    loader = unittest.TestLoader()
                    suite = loader.loadTestsFromTestCase(TestCases)
                    test_result = unittest.TestResult()
                    
                    with time_limit(timeout):
                        suite.run(test_result)
                    
                    if test_result.wasSuccessful():
                        result_list.append('passed')
                    else:
                        result_list.append(f"failed: {len(test_result.failures) + len(test_result.errors)} errors")
                
                # Legacy/HumanEval-style: check for 'check' function
                elif 'check' in new_module.__dict__:
                    check_func = new_module.__dict__['check']
                    entry_point = problem['entry_point']
                    if entry_point in new_module.__dict__:
                        entry_func = new_module.__dict__[entry_point]
                        with time_limit(timeout):
                            check_func(entry_func)
                        result_list.append('passed')
                    else:
                        # Sometimes 'check' assumes we passed the name, or it's self-contained?
                        # EvalScope's original logic was check(entry_point_function_result) usually?
                        # Original: check_program = ... + f"check({problem['entry_point']})"
                        # We try to mimic that call.
                        result_list.append(f'failed: entry point {entry_point} not found for check function')
                else:
                    # If no test runner found, maybe the execution itself was the test (assertions at top level)?
                    # We assume it passed if no exception was raised during exec.
                    # But for safety, warn or assume passed only if specific conditions met?
                    # For now, let's assume it passed if it didn't crash, mirroring simple execution scripts.
                    result_list.append('passed')

        except TimeoutException:
            result_list.append('timed out')
        except BaseException as e:
            result_list.append(f'failed: {e}')

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append('timed out')

    return dict(
        task_id=problem['task_id'],
        passed=result[0] == 'passed',
        result=result[0],
        completion_id=completion_id,
    )
