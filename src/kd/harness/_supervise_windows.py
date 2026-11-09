
from __future__ import annotations

import contextlib
import ctypes
import msvcrt
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, Literal

import psutil


assert sys.platform == "win32"

_GIB: Final = 1024**3



_TERMINATED_EXIT_CODE: Final = 1

_CREATE_SUSPENDED: Final = 0x00000004
_CREATE_UNICODE_ENVIRONMENT: Final = 0x00000400



_CREATE_NO_WINDOW: Final = 0x08000000
_STARTF_USESTDHANDLES: Final = 0x00000100

_JOB_OBJECT_LIMIT_JOB_MEMORY: Final = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
_JOB_OBJECT_BASIC_PROCESS_ID_LIST: Final = 3
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Final = 9

_MEMBER_CAPACITY: Final = 1024

_WAIT_OBJECT_0: Final = 0x00000000
_WAIT_FAILED: Final = 0xFFFFFFFF
_RESUME_FAILED: Final = 0xFFFFFFFF

_HANDLE = ctypes.c_void_p
_BOOL = ctypes.c_int32
_WORD = ctypes.c_uint16
_DWORD = ctypes.c_uint32
_SIZE_T = ctypes.c_size_t
_LARGE_INTEGER = ctypes.c_int64


class _StartupInfo(ctypes.Structure):

    _fields_ = [
        ("cb", _DWORD),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", _DWORD),
        ("dwY", _DWORD),
        ("dwXSize", _DWORD),
        ("dwYSize", _DWORD),
        ("dwXCountChars", _DWORD),
        ("dwYCountChars", _DWORD),
        ("dwFillAttribute", _DWORD),
        ("dwFlags", _DWORD),
        ("wShowWindow", _WORD),
        ("cbReserved2", _WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", _HANDLE),
        ("hStdOutput", _HANDLE),
        ("hStdError", _HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):

    _fields_ = [
        ("hProcess", _HANDLE),
        ("hThread", _HANDLE),
        ("dwProcessId", _DWORD),
        ("dwThreadId", _DWORD),
    ]


class _BasicLimitInformation(ctypes.Structure):

    _fields_ = [
        ("PerProcessUserTimeLimit", _LARGE_INTEGER),
        ("PerJobUserTimeLimit", _LARGE_INTEGER),
        ("LimitFlags", _DWORD),
        ("MinimumWorkingSetSize", _SIZE_T),
        ("MaximumWorkingSetSize", _SIZE_T),
        ("ActiveProcessLimit", _DWORD),
        ("Affinity", _SIZE_T),
        ("PriorityClass", _DWORD),
        ("SchedulingClass", _DWORD),
    ]


class _IoCounters(ctypes.Structure):

    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtendedLimitInformation(ctypes.Structure):

    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", _SIZE_T),
        ("JobMemoryLimit", _SIZE_T),
        ("PeakProcessMemoryUsed", _SIZE_T),
        ("PeakJobMemoryUsed", _SIZE_T),
    ]


class _ProcessIdList(ctypes.Structure):

    _fields_ = [
        ("NumberOfAssignedProcesses", _DWORD),
        ("NumberOfProcessIdsInList", _DWORD),
        ("ProcessIdList", _SIZE_T * _MEMBER_CAPACITY),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _declare(name: str, restype: Any, *argtypes: Any) -> Any:
    function = getattr(_kernel32, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


_create_job_object = _declare(
    "CreateJobObjectW", _HANDLE, ctypes.c_void_p, ctypes.c_wchar_p
)
_set_information_job_object = _declare(
    "SetInformationJobObject",
    _BOOL,
    _HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    _DWORD,
)
_query_information_job_object = _declare(
    "QueryInformationJobObject",
    _BOOL,
    _HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    _DWORD,
    ctypes.POINTER(_DWORD),
)
_assign_process_to_job_object = _declare(
    "AssignProcessToJobObject", _BOOL, _HANDLE, _HANDLE
)
_terminate_job_object = _declare("TerminateJobObject", _BOOL, _HANDLE, ctypes.c_uint)
_create_process = _declare(
    "CreateProcessW",
    _BOOL,
    ctypes.c_wchar_p,
    ctypes.c_wchar_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    _BOOL,
    _DWORD,
    ctypes.c_void_p,
    ctypes.c_wchar_p,
    ctypes.POINTER(_StartupInfo),
    ctypes.POINTER(_ProcessInformation),
)
_resume_thread = _declare("ResumeThread", _DWORD, _HANDLE)
_wait_for_single_object = _declare("WaitForSingleObject", _DWORD, _HANDLE, _DWORD)
_get_exit_code_process = _declare(
    "GetExitCodeProcess", _BOOL, _HANDLE, ctypes.POINTER(_DWORD)
)
_terminate_process = _declare("TerminateProcess", _BOOL, _HANDLE, ctypes.c_uint)
_close_handle = _declare("CloseHandle", _BOOL, _HANDLE)


def _win_error(call: str, filename: str | None = None) -> OSError:
    code = ctypes.get_last_error()
    return OSError(0, f"{call}: {ctypes.FormatError(code).strip()}", filename, code)


def _create_job(memory_limit_bytes: int | None) -> int:
    job: int | None = _create_job_object(None, None)
    if job is None:
        raise _win_error("CreateJobObjectW")
    limits = _ExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if memory_limit_bytes is not None:
        limits.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
        limits.JobMemoryLimit = memory_limit_bytes
    if not _set_information_job_object(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        error = _win_error("SetInformationJobObject")
        _close_handle(job)
        raise error
    return job


def _inheritable_copy(stack: contextlib.ExitStack, fd: int) -> int:
    copy = os.dup(fd)
    stack.callback(os.close, copy)
    os.set_inheritable(copy, True)
    return msvcrt.get_osfhandle(copy)


def _environment_block(env: Mapping[str, str]) -> ctypes.Array[ctypes.c_wchar]:

    entries = "".join(f"{name}={value}\0" for name, value in env.items())
    return ctypes.create_unicode_buffer(entries or "\0")


def _launch(
    job: int,
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path | None,
    stdout: IO[Any] | None,
    stderr: IO[Any] | None,
) -> tuple[int, int]:
    startup = _StartupInfo()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = _STARTF_USESTDHANDLES
    created = _ProcessInformation()


    try:
        with contextlib.ExitStack() as stack:
            null = os.open(os.devnull, os.O_RDONLY)
            stack.callback(os.close, null)
            startup.hStdInput = _inheritable_copy(stack, null)
            startup.hStdOutput = _inheritable_copy(
                stack, 1 if stdout is None else stdout.fileno()
            )
            startup.hStdError = _inheritable_copy(
                stack, 2 if stderr is None else stderr.fileno()
            )
            if not _create_process(
                None,
                ctypes.create_unicode_buffer(subprocess.list2cmdline(command)),
                None,
                None,
                True,
                _CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT | _CREATE_NO_WINDOW,
                _environment_block(env),
                None if cwd is None else str(cwd),
                ctypes.byref(startup),
                ctypes.byref(created),
            ):
                raise _win_error("CreateProcessW", command[0])

        if not _assign_process_to_job_object(job, created.hProcess):
            raise _win_error("AssignProcessToJobObject")
        if _resume_thread(created.hThread) == _RESUME_FAILED:
            raise _win_error("ResumeThread")
    except BaseException:

        if created.hProcess is not None:
            _terminate_process(created.hProcess, _TERMINATED_EXIT_CODE)
            _close_handle(created.hProcess)
        raise
    finally:
        if created.hThread is not None:
            _close_handle(created.hThread)
    return created.hProcess, created.dwProcessId


def check_memory_cap_available() -> None:
    with open(os.devnull, "wb") as sink:
        probe = ProcessTree(
            ["cmd.exe", "/c", "exit"],
            env=os.environ,
            cwd=None,
            stdout=sink,
            stderr=sink,
            memory_max_gb=1.0,
        )
    try:
        probe.wait_leader(30.0)
        returncode = probe.leader_returncode()
    finally:
        probe.close()
    if returncode != 0:
        raise RuntimeError(
            "a command in a memory-capped Job Object did not complete here: "
            f"exit status {returncode}"
        )


class ProcessTree:

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None,
        stdout: IO[Any] | None,
        stderr: IO[Any] | None,
        memory_max_gb: float | None,
    ) -> None:
        self.memory_backend: Literal["windows-job"] | None = (
            None if memory_max_gb is None else "windows-job"
        )
        self.memory_limit_bytes: int | None = (
            None if memory_max_gb is None else int(memory_max_gb * _GIB)
        )
        self._returncode: int | None = None
        self._job: int | None = _create_job(self.memory_limit_bytes)
        try:
            process, self.pid = _launch(
                self._job, command, env=env, cwd=cwd, stdout=stdout, stderr=stderr
            )
        except BaseException:
            _close_handle(self._job)
            raise
        self._process: int | None = process

    def leader_returncode(self) -> int | None:
        if self._returncode is None and self._leader_exited(0):
            status = _DWORD()
            if not _get_exit_code_process(self._process, ctypes.byref(status)):
                raise _win_error("GetExitCodeProcess")
            self._returncode = status.value
        return self._returncode

    def _leader_exited(self, milliseconds: int) -> bool:
        waited: int = _wait_for_single_object(self._process, milliseconds)
        if waited == _WAIT_FAILED:
            raise _win_error("WaitForSingleObject")
        return waited == _WAIT_OBJECT_0

    def _member_pids(self) -> list[int]:
        members = _ProcessIdList()
        if not _query_information_job_object(
            self._job,
            _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            ctypes.byref(members),
            ctypes.sizeof(members),
            None,
        ):
            raise _win_error("QueryInformationJobObject")
        return list(members.ProcessIdList[: members.NumberOfProcessIdsInList])

    def members_remain(self) -> bool:
        return any(pid != self.pid for pid in self._member_pids())

    def rss_bytes(self) -> int:
        total = 0
        for pid in self._member_pids():

            with contextlib.suppress(psutil.NoSuchProcess):
                total += psutil.Process(pid).memory_info().rss
        return total

    def peak_committed_bytes(self) -> int | None:
        limits = _ExtendedLimitInformation()
        if not _query_information_job_object(
            self._job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            None,
        ):
            raise _win_error("QueryInformationJobObject")
        return int(limits.PeakJobMemoryUsed)

    def stop(self, *, force: bool) -> None:
        if force and not _terminate_job_object(self._job, _TERMINATED_EXIT_CODE):
            raise _win_error("TerminateJobObject")

    def wait_leader(self, seconds: float) -> None:
        self._leader_exited(round(seconds * 1000))

    def close(self) -> None:
        if self._job is None:
            return
        handles = (self._process, self._job)
        self._process = self._job = None
        closed = [bool(_close_handle(handle)) for handle in handles]
        if not all(closed):
            raise _win_error("CloseHandle")
