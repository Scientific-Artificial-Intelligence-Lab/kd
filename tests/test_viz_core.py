from pathlib import Path

import pytest

from kd.viz import core as viz_core
from kd.viz import registry as viz_registry


class DummyModel:
    pass


class DummyAdapter:
    capabilities = {'example'}

    def __init__(self):
        self.calls = []

    def render(self, request, ctx):
        path = ctx.save_path('dummy.txt')
        self.calls.append({'kind': request.kind, 'path': path})
        return viz_core.VizResult(intent=request.kind, paths=[path])


def setup_function():
    viz_registry.clear_registry()


def teardown_function():
    viz_registry.clear_registry()


def test_render_without_adapter():
    request = viz_core.VizRequest(kind='example', target=DummyModel())
    result = viz_core.render(request)
    assert not result.has_content
    assert result.warnings
    assert 'No visualization adapter' in result.warnings[0]


def test_render_with_registered_adapter(tmp_path):
    adapter = DummyAdapter()
    viz_registry.register_adapter(DummyModel, adapter)

    viz_core.configure(save_dir=tmp_path)
    try:
        request = viz_core.VizRequest(
            kind='example',
            target=DummyModel(),
        )
        result = viz_core.render(request)
    finally:
        viz_core.configure(save_dir=None)

    assert adapter.calls
    call = adapter.calls[0]
    assert call['kind'] == 'example'
    assert result.paths == [tmp_path / 'dummy.txt']


def test_render_with_unsupported_intent():
    class NarrowAdapter:
        capabilities = {'other'}

        def render(self, request, ctx):  # pragma: no cover - not expected to run
            raise AssertionError

    viz_registry.register_adapter(DummyModel, NarrowAdapter())
    request = viz_core.VizRequest(kind='example', target=DummyModel())

    result = viz_core.render(request)
    assert not result.has_content
    assert 'does not support intent' in result.warnings[0]
    assert result.metadata['capabilities'] == ['other']


def test_list_capabilities_reflects_registry():
    adapter = DummyAdapter()
    viz_registry.register_adapter(DummyModel, adapter)

    caps = viz_core.list_capabilities(DummyModel())
    assert set(caps) == {'example'}

    viz_registry.unregister_adapter(DummyModel)
    assert list(viz_core.list_capabilities(DummyModel())) == []
