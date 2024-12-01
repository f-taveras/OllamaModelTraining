from unsloth import FastLanguageModel
import torch
import os
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformets import TrainingArguments
from unsloth import is_bfloat16_supported





# configuration 


max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.


### Instruction:

{}

### Input:
{}

### Response:
{}"""

instruction = "determine weather the answer is correct or not, only answer yes, or no"
input = "2+2=4"


# before training 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.getenv("HF_TOKEN"),
)

FastLanguageModel.for_inference(model) # enable native 2x faster inference NOT SURE WHAT That means
inputs = tokenizer(
[
    alpaca_prompt.format(
        instruction,
        input,
        "",
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)


# Load data

EOS_TOKEN = tokenizer.eos_token # must add EOS TOCKEN, WHATEVER THAT MEANS 
def formatting_prompts_func(examples):
    instructions = examples["instructions"]
    inputs = examples["inputs"]
    outputs = examples["outputs"]
    texts = []
    for instriction, input, output in zip(instructions, inputs, outputs):
        texts = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(texts)
    return {"text": texts, }

pass
dataset = load_dataset("iamtarun/python_code_instructoins_18k_alpaca", split = "train")
dataset = dataset.map(formatting_prompts_func, batch = True,  )


# training 


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #Choose any number > 0! suggested 8, 16, 32 , 64 not sure what this is 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", ],
    lora_alph = 16,
    lora_dropout = 0, # supports any, but = 0 is optimized??
    bias = "none", # supports any, but = none is optimized??
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None

)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, #can make training 5x faster for short sequences, vs what?
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run
        
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw-8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "output",
    ),
)



# Memory stats display (optional, just to monitor memory usage)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Show final memory and time stats (optional, just to monitor memory usage)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# After Training 


FastLanguageModel.for_interference(model) # enable native 2x faster inference (vs what?)
input = tokenizer(
    [
        alpaca_prompt.format(
            instruction,
            input,
            "",
            
        )
    ], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)

# 6. Saving
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
model.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN")) 
tokenizer.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN"))

# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))