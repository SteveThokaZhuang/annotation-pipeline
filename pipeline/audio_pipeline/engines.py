"""Release vLLM's owned subprocesses before the worker interpreter exits."""


def shutdown_vllm(engine):
    if getattr(engine,'_pipeline_closed',False):return
    engine._pipeline_closed=True
    closer=getattr(engine,'shutdown',None)
    if closer is None:
        core=getattr(getattr(engine,'llm_engine',None),'engine_core',None)
        closer=getattr(core,'shutdown',None)
    if closer is not None:closer()
