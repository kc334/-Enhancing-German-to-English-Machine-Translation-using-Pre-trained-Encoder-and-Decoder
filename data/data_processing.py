from datasets import load_dataset
def write_sentences_to_file(dataset, language, output_file):
    sentences = dataset['translation']
    with open(output_file, 'w') as file:
        for sentence in sentences:
            print(sentence[language])
            file.write(sentence[language] + '\n')

# Assuming you have your dataset loaded in a variable called 'datasets'
dataset = load_dataset("bbaaaa/iwslt14-de-en")
# Writing 'de' sentences to a file
write_sentences_to_file(dataset['validation'], 'de', './dev.txt')
write_sentences_to_file(dataset['test'], 'de', './test.txt')
# Writing 'en' sentences to a file

# write_sentences_to_file(datasets['validation'], 'en', 'dev.out')
