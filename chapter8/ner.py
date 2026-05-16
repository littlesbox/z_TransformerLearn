# from transformers import pipeline

# token_classifier = pipeline("token-classification", device=0)
# results = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# print(results)




from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="max")
results = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(results)
