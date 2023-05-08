import torch
import np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class Covid19StanceDatasetCP(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.span = [t.index(30522) for t in self.encodings['input_ids']]

        print(len(self.span))
        print(len(self.labels))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['e_span'] = torch.tensor(self.span[idx])
        return item

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def pad_input(input_encodings, pad_val=0, max_len=128):
        return [(np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=(pad_val))).tolist() for seq in
                input_encodings]

    @staticmethod
    def encode_cure_prevention_with_max_len(texts, cure_prevention, tokenizer, max_len=128):
        # find cp position in text
        new_texts = []
        for text, cp in zip(texts, cure_prevention):
            if len(tokenizer.tokenize(text)) <= max_len:
                start_idx = text.find(cp)
                p1 = text[:start_idx]
                p2 = text[start_idx:start_idx + len(cp)]
                p3 = text[start_idx + len(cp):]

                new_p = p1 + '<cp_start>' + p2 + '<cp_end>' + p3
                new_texts.append(new_p)
            else:
                start_idx = text.find(cp)
                p1 = text[:max_len]
                p2 = text[start_idx:start_idx + len(cp)]
                p3 = text[start_idx + len(cp):]

                new_p = p1 + '<cp_start>' + p2 + '<cp_end>'
                new_texts.append(new_p)

        return new_texts

    @staticmethod
    def get_processed_data(texts, labels, cp, tokenizer, ratio):
        train_text, test_text, train_label, test_label, train_cp, test_cp = train_test_split(texts, labels, cp,
                                                                                             test_size=1 - ratio,
                                                                                             random_state=0)
        test_text, dev_text, test_label, dev_label, test_cp, dev_cp = train_test_split(test_text, test_label,
                                                                                          test_cp,
                                                                                          test_size=0.5,
                                                                                          random_state=0)

        train_text_cp = Covid19StanceDatasetCP.encode_cure_prevention_with_max_len(train_text, train_cp, tokenizer)
        dev_text_cp = Covid19StanceDatasetCP.encode_cure_prevention_with_max_len(dev_text, dev_cp, tokenizer)
        test_text_cp = Covid19StanceDatasetCP.encode_cure_prevention_with_max_len(test_text, test_cp, tokenizer)

        train_encodings_cp = tokenizer(train_text_cp, truncation=True, max_length=128)
        dev_encodings_cp = tokenizer(dev_text_cp, truncation=True, max_length=128)
        test_encodings_cp = tokenizer(test_text_cp, truncation=True, max_length=128)

        # manually pad input
        train_encodings_cp['input_ids'] = Covid19StanceDatasetCP.pad_input(train_encodings_cp['input_ids'])
        dev_encodings_cp['input_ids'] = Covid19StanceDatasetCP.pad_input(dev_encodings_cp['input_ids'])
        test_encodings_cp['input_ids'] = Covid19StanceDatasetCP.pad_input(test_encodings_cp['input_ids'])

        train_encodings_cp['token_type_ids'] = Covid19StanceDatasetCP.pad_input(train_encodings_cp['token_type_ids'])
        dev_encodings_cp['token_type_ids'] = Covid19StanceDatasetCP.pad_input(dev_encodings_cp['token_type_ids'])
        test_encodings_cp['token_type_ids'] = Covid19StanceDatasetCP.pad_input(test_encodings_cp['token_type_ids'])

        train_encodings_cp['attention_mask'] = Covid19StanceDatasetCP.pad_input(train_encodings_cp['attention_mask'])
        dev_encodings_cp['attention_mask'] = Covid19StanceDatasetCP.pad_input(dev_encodings_cp['attention_mask'])
        test_encodings_cp['attention_mask'] = Covid19StanceDatasetCP.pad_input(test_encodings_cp['attention_mask'])

        train_dataset_cp = Covid19StanceDatasetCP(train_encodings_cp, train_label)
        dev_dataset_cp = Covid19StanceDatasetCP(dev_encodings_cp, dev_label)
        test_dataset_cp = Covid19StanceDatasetCP(test_encodings_cp, test_label)

        return train_dataset_cp, dev_dataset_cp, test_dataset_cp, train_label, dev_label, test_label