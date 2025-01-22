import ldp.nn as ldpnn

# if TEST_GPUS:
#     # TODO: figure out how to run CPU tests when a GPU is available.
#     # Our code uses LOCAL_RANK everywhere to determine device placement,
#     # so changing the model location just causes a mismatch. Also setting
#     # CUDA_VISIBLE_DEVICES after torch import has no effect, so we can't
#     # change it on the fly.
#     FSDP_ENABLED = [False, True]
#     DEVICES = ["cuda:0"]
#     # mock torchrun env vars
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "8999"
#     os.environ["LOCAL_RANK"] = os.environ["RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
# else:
#     FSDP_ENABLED = [False]
#     DEVICES = ["cpu"]


def test_model_load():
    config = ldpnn.LMConfig(
        model="gpt2",
        load_args={"attn_implementation": "eager"},
        dtype=ldpnn.TorchDType.fp32,
    )
    config.get_causal_lm()
