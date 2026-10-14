# Checks both_reader.get's retry in Pyodide: a fake fetch answers with given statuses, then the real one runs once.
import asyncio
import time

import both_reader as R

real = R.pyfetch


class Fake:
    def __init__(self, status, headers=None):
        self.status, self.headers = status, headers or {}

    async def bytes(self):
        return b""


def script(*answers):
    """Each answer: a status (fake response), an exception (fetch fails), or None (the real fetch)."""
    seq = list(answers)

    async def fetch(url, headers):
        a = seq.pop(0)
        if a is None:
            return await real(url, headers=headers)
        if isinstance(a, Exception):
            raise a
        return Fake(*a) if isinstance(a, tuple) else Fake(a)
    return fetch


URL, RNG = f"{R.AEF}/2023/16N/x4v3mzlodlbp170gg-0000000000-0000000000.tiff", "bytes=0-1023"


async def case(name, answers, expect):
    R.pyfetch = script(*answers)
    R.stats.update(gets=0, bytes=0, retries=0)
    t = time.time()
    try:
        b = await R.get(URL, RNG)
        got = f"{len(b)} bytes" if b is not None else "None"
    except OSError as e:
        got = f"OSError {str(e)[:60]}"
    ok = got.startswith(expect)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {got}, retries {R.stats['retries']}, {time.time() - t:.1f} s")
    return ok


async def main():
    res = [
        await case("two 503s then real", [503, 503, None], "1024 bytes"),
        await case("429 with Retry-After 1", [(429, {"retry-after": "1"}), None], "1024 bytes"),
        await case("fetch fails then real", [OSError("Failed to fetch"), None], "1024 bytes"),
        await case("403 is not retried", [403], "OSError 403"),
        await case("404 is None", [404], "None"),
        await case("503 six times gives up", [503] * 6, "OSError 503"),
    ]
    R.pyfetch = real
    print("RESULT", "all pass" if all(res) else "FAIL")

await main()
