import os
import polars as pl
import asyncio
import art
from art.local import LocalBackend
from openpipe.client import AsyncOpenPipe
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Config ---
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_NAME = "hotpotqa-grpo-lr3e5-gpu085"
LR = 3e-5
GPU_UTIL = 0.85
MAX_TOKENS = 256
CTX_LEN = 2048
RL_STEPS = 10

# Load API keys
load_dotenv()
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
os.environ["OPENPIPE_API_KEY"] = "YOUR_OPENPIPE_API_KEY"

# Load dataset
df = pl.read_parquet("./Agent-R1/data/hotpotqa/train.parquet")

class HotpotSample(BaseModel):
    prompt: str
    ground_truth: str

def random_sample(n=16):
    samples = df.sample(n=n).to_dicts()
    return [
        HotpotSample(
            prompt=ex['prompt'][0]['content'],
            ground_truth=ex['reward_model']['ground_truth']
        ) for ex in samples
    ]

# Model config
model = art.TrainableModel(
    name=RUN_NAME,
    project="hotpotqa-grpo-experiments",
    base_model=BASE_MODEL,
)

model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=CTX_LEN,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=GPU_UTIL,
        num_scheduler_steps=5,
    ),
)

# Backend + client
backend = LocalBackend(in_process=True, path="./.art")
op_client = AsyncOpenPipe()

@art.retry()
async def rollout(model: art.Model, sample: HotpotSample) -> art.Trajectory:
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Answer the question using multi-hop reasoning."},
            {"role": "user", "content": sample.prompt},
        ],
        reward=0,
    )
    messages = trajectory.messages()
    completion = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model.name,
        max_completion_tokens=MAX_TOKENS,
    )
    choice = completion.choices[0]
    generated = choice.message.content
    trajectory.messages_and_choices.append(choice)

    gt = sample.ground_truth.strip().lower()
    gen = generated.strip().lower()
    trajectory.reward = 1 if gt in gen else 0

    if op_client.api_key:
        await op_client.report(
            requested_at=0, received_at=0,
            req_payload={"model": model.name, "messages": messages},
            resp_payload=completion,
            status_code=200,
        )
    return trajectory

async def train():
    await model.register(backend)
    for step in range(await model.get_step(), RL_STEPS):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, sample) for sample in random_sample(16)
                )
                for _ in range(1)
            ),
            pbar_desc=f"{RUN_NAME} Step {step}",
        )
        await model.delete_checkpoints()
        await model.train(
            train_groups,
            config=art.TrainConfig(
                learning_rate=LR,
            ),
            _config={"logprobcalculation_chunk_size": 8},
        )
    print("Training completed.")

if __name__ == "__main__":
    asyncio.run(train())
