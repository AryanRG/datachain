import asyncio
import io

from datachain.client.listing_worker import _serve, dump, load


class FakeClient:
    def __init__(self, out, fail=False):
        self.out, self.fail, self.probed_before_ready = out, fail, None

    async def _probe(self, prefix, start_after):
        self.probed_before_ready = not self.out.getvalue()
        if self.fail:
            raise PermissionError("no list access")


def serve(client, out):
    inp = io.BytesIO()  # no requests: the child stops at end of input
    asyncio.run(_serve(client, inp, out))
    out.seek(0)
    return load(out)


def test_child_warms_up_before_it_reports_ready():
    out = io.BytesIO()
    client = FakeClient(out)

    assert serve(client, out) == ("ready", None)
    assert client.probed_before_ready


def test_child_reports_ready_when_warm_up_fails():
    out = io.BytesIO()

    assert serve(FakeClient(out, fail=True), out) == ("ready", None)


def test_frames_round_trip():
    assert load(io.BytesIO(dump(("page", 1, [("s3://b", "k")], "k", "k")))) == (
        "page",
        1,
        [("s3://b", "k")],
        "k",
        "k",
    )
