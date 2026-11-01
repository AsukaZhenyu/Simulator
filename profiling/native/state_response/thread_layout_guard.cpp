// Compile-only verification against the explicitly selected Windows SDK.
#include <Windows.h>
#include <winternl.h>
#include <cstddef>
static_assert(sizeof(void*) == 8, "thread snapshot requires x64");
static_assert(sizeof(SYSTEM_PROCESS_INFORMATION) == 256, "process header layout changed");
static_assert(offsetof(SYSTEM_PROCESS_INFORMATION, NextEntryOffset) == 0, "next offset changed");
static_assert(offsetof(SYSTEM_PROCESS_INFORMATION, NumberOfThreads) == 4, "thread count offset changed");
static_assert(offsetof(SYSTEM_PROCESS_INFORMATION, UniqueProcessId) == 80, "process id offset changed");
static_assert(sizeof(SYSTEM_THREAD_INFORMATION) == 80, "thread header layout changed");
static_assert(offsetof(SYSTEM_THREAD_INFORMATION, ClientId) == 40, "thread identity offset changed");
static_assert(offsetof(SYSTEM_THREAD_INFORMATION, ThreadState) == 68, "thread state offset changed");
static_assert(offsetof(SYSTEM_THREAD_INFORMATION, WaitReason) == 72, "wait reason offset changed");
