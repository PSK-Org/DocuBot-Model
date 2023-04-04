import hydra
from omegaconf import OmegaConf
import logging
import qa
import load
from dbus_next.aio.message_bus import MessageBus
from dbus_next import Variant
import asyncio
import torch_xla.core.xla_model as xm

@hydra.main(config_path='config', config_name='config')
def run(config):
    print("<config>")
    print(OmegaConf.to_yaml(config))
    print("<\config>")
    
    # set device
    
    device = config.device

    if config.is_using_tpu:
        device = xm.xla_device()
        
    # logging
        
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)
    
    # Load Data
    
    doc_dir, document_store = load.init_store(config)
    
    load.download_data(config, doc_dir)
    
    load.load_doc(config, document_store, doc_dir)
    
    ai = qa.EditAI(config.dbus.ai_interface, config, document_store)
    
    if config.mode == "launch":
        # Setup QA Pipeline

        asyncio.get_event_loop().run_until_complete(setup_bus(config, ai))
        
    elif config.mode == "test":
            
            ai.test()
    
    else:
        
        raise ValueError(f"config.mode: {config.mode} is invalid")
    
    
async def setup_bus(config, ai):
    
    # setup bus
    
    bus = await MessageBus().connect()
    
    bus.export(config.dbus.ai_object_path, ai)
    
    await bus.request_name(config.dbus.bus_name)
    
    print("AI is online")
    
    await bus.wait_for_disconnect()

    
if __name__ == "__main__":
    run()