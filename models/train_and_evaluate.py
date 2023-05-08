import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train_cp(train_dataset, dev_dataset, model, lr=3e-6, bs=8, warmup_ratio=0.1, num_epochs=7):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.cuda()
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    num_train_steps = int(len(train_dataset) / bs * num_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=int(warmup_ratio * num_train_steps),
                                                num_training_steps=num_train_steps)

    best_f1 = 0
    best_model = None
    accum_iter = 8  # gradient accumulation
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        model.train()
        count = 0
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # print(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, e_span=batch['e_span'])

            loss = outputs[0]
            loss = loss / accum_iter  # gradient accumulation

            total_loss += loss.item()
            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):  # gradient accumulation
                optim.step()
                scheduler.step()
                optim.zero_grad()
            count += bs
        avg_train_loss = total_loss / len(train_loader)
        print("Average train loss: {}".format(avg_train_loss))
        f1 = evaluate_cp(model, dev_dataset)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    return best_model, best_f1


def evaluate_cp(model, dev_dataset, bs = 4):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False)

    golds = []
    preds = []
    nb_eval_steps, eval_loss = 0, 0

    for batch in dev_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, e_span=batch['e_span'])
        eval_loss += outputs[0].mean().item()
        golds.extend(labels.cpu().detach().numpy().reshape(-1))
        preds.extend(torch.argmax(outputs[1], dim=-1).cpu().detach().numpy())
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))

    print(classification_report(golds, preds, target_names=['Agree', 'Disagree', 'No Stance'], digits=4))
    f1 = f1_score(golds, preds, average='macro')
    return f1