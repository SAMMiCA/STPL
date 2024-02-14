from transformers import pipeline, set_seed

set_seed(42)

generator = pipeline('text-generation', model='./distilgpt_results_new_e5', tokenizer='distilgpt2')

prompt = ("### Human: what's an appropriate subtask sequence to achieve the following task?: "
          "pick up alarm clock on countertop then place it in cabinet "
          "Only use the following words: "
          "[teleport to, open, pick up, close, place, in , on, countertop, alarm clock, cabinet] "
          "### Assistant: 1. teleport to countertop 2. pick up alarm clock 3. teleport to cabinet 4. open cabinet "
          "5. place alarm clock in cabinet 6. close cabinet")


prompt += ("### Human: what's an appropriate subtask sequence to achieve the following task?: pick up boots in cabinet then place it in box Only use the following words: [teleport to, open, pick up, close, place, in , on, cabinet, boots, box] ### Assistant: 1. ")
output = generator(prompt, max_length=220, num_return_sequences=5)
print(output[0])
print('\n')
print(output[1])
print('\n')
print(output[2])
print('\n')
print(output[3])
print('\n')
print(output[4])

