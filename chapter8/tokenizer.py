from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)


tokens = encoding.tokens()
# print(tokens)
word_dict = {}
for i in range(len(tokens)):
    # print(i)
    word_index = encoding.token_to_word(i)
    # print(word_index)
    if word_index is not None:
        start, end = encoding.word_to_chars(word_index)
        word = example[start:end]
        if word not in word_dict:
            word_dict[word] = 1
            print(word)

# a = 0 is None
# print(a)



# token_index = 5
# print('the 5th token is:', encoding.tokens()[token_index])
# start, end = encoding.token_to_chars(token_index)
# print('corresponding text span is:', example[start:end])
# word_index = encoding.word_ids()[token_index] # 3
# start, end = encoding.word_to_chars(word_index)
# print('corresponding word span is:', example[start:end])



# token_index = 5
# print('the 5th token is:', encoding.tokens()[token_index])
# corresp_word_index = encoding.token_to_word(token_index)
# print('corresponding word index is:', corresp_word_index)
# start, end = encoding.word_to_chars(corresp_word_index)
# print('the word is:', example[start:end])
# start, end = encoding.word_to_tokens(corresp_word_index)
# print('corresponding tokens are:', encoding.tokens()[start:end])



# chars = 'My name is Sylvain'
# print('characters of "{}" ars: {}'.format(chars, list(chars)))
# print('corresponding word index: ')
# for i, c in enumerate(chars):
#     print('"{}": {} '.format(c, encoding.char_to_word(i)), end="")
# print('\ncorresponding token index: ')
# for i, c in enumerate(chars):
#     print('"{}": {} '.format(c, encoding.char_to_token(i)), end="")