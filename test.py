from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to("cuda")

# Prepare the input to the model
prompt = "Your task is to translate between English and the low resource language of Central Aymara. Here are some examples:\n\nExample 1:\n\nEnglish: We got up a five a.m. and took a 3 mile jaunt all around our neighborhood. It was great exercise and a great way to start the day! How did your day start today?\nCentral Aymara: Phisqha alwa pachaw sartapxta ukatx utaj jak’an 3 millas ukaruw muytir sarapxta. Jichhürux kusapuniw ukham muytir sarañaxa! Jumatakist kunjamakis jichhüruxa?\n\nExample 2:\n\nEnglish: It would be awesome to go to one of your shows! Do you have anything coming up?\nCentral Aymara: Walikipuniw jutawa! Jutir urutak utjtamti?\n\nNow your turn. Translate the following English sentence into Aymara:\n\nEnglish: What's the weather like in Arequipa today?\nCentral Aymara: "
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Set to False to strictly disable thinking
)
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")

# Generate outputs
outputs = model.generate(
    input_ids,
    num_return_sequences=32,
    do_sample=True,
)

flat_sequences = outputs.reshape(32, 1024)

# Prepare decoder inputs/targets
decoder_input_ids = flat_sequences[:, :-1]
target_ids = flat_sequences[:, 1:]
log_probs = model(input_ids=input_ids, labels=target_ids)
logits = log_probs.logits
print(logits.shape)