import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import transformers
from torch.utils.tensorboard import SummaryWriter
import os
import json
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--input_data_folder', type=str, help='Input folder')

    args = parser.parse_args()
    MODEL_PATH = 'xlm-roberta-base'
    BATCH_SIZE = 16
    N_EPOCHS = 1
    LR = 1e-5
    LOAD_CHECKPOINT = False #�'{args.input_data_folder}/model_checkpoint.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RobertaRegressorTextOnly(n_regression = 10, model_path=MODEL_PATH ).to(device)

    if LOAD_CHECKPOINT:
        print(F'Loaded checkpoint: {LOAD_CHECKPOINT}')
        model.load_state_dict(torch.load(LOAD_CHECKPOINT))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    output_folder = args.output_folder

    train_dataset = InfluencerDatasetTextOnly(
        f'{args.input_data_folder}/X_train.csv',
        f'{args.input_data_folder}/ys_train.csv',
        MODEL_PATH
        )

    val_dataset = InfluencerDatasetTextOnly(
        f'{args.input_data_folder}/X_val.csv',
        f'{args.input_data_folder}/ys_val.csv',
        MODEL_PATH
        )

    test_dataset = InfluencerDatasetTextOnly(
        f'{args.input_data_folder}/X_test.csv',
        f'{args.input_data_folder}/ys_test.csv',
        MODEL_PATH
        )

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    writer = SummaryWriter(f'{output_folder}/logs')
    output_folder = '.'
    last_loss = 0

    # Train and validate
    for epoch_index in range(N_EPOCHS):

        running_loss = 0.0
        print(f'Epochs {epoch_index + 1}/{N_EPOCHS}')
        for i, data in enumerate(training_loader):
            model.train()
            # Every data instance is an input + label pair
            inputs, labels = data
            labels = labels.type(torch.float)

            optimizer.zero_grad()
            outputs = model(inputs).cpu()
            loss = myLoss(outputs, labels)

            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if i % 3000 == 0:
                model.eval()
                torch.save(model.state_dict(), f'{output_folder}/checkpoints-model-LR-{LR}.pt')
                model.train()
                print(f'Iter {i + 1}, train batch loss: {round(running_loss / ((i + 1) * BATCH_SIZE), 5)}')
                writer.add_scalar("Loss/train", running_loss / ((i + 1) * BATCH_SIZE), i * (epoch_index + 1))

        print(f'Train batch loss: {round(running_loss / ((i + 1) * BATCH_SIZE), 5)}')
        writer.add_scalar("Loss/train", running_loss / ((i + 1) * BATCH_SIZE), i*(epoch_index+1))

        model.eval()
        l1loss = 0.0

        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                outputs = model(inputs)
                labels = labels.type(torch.float).to(device)
                loss = myLoss(outputs, labels)
                l1loss += loss.item()

            if l1loss < last_loss:
                last_loss = l1loss
                print(f"Saving checkpoint {epoch_index}")
                model.eval()
                torch.save(model.state_dict(), f'{output_folder}/best-model-val-LR-{LR}.pt')
                model.train()
        print()

    test_predictions = []
    test_labels = []

    # Test
    with torch.no_grad():
      for j, data in tqdm(enumerate(test_loader)):
        inputs, labels = data
        outputs = model(inputs).to('cpu')
        labels = labels.type(torch.float)
        test_predictions.append(outputs.numpy())
        test_labels.append(labels.numpy())
        
        
    test_predictions = np.concatenate(test_predictions, axis = 0)
    test_labels = np.concatenate(test_labels, axis = 0)

    reactions = ['angryCount', 'careCount', 'hahaCount',
       'likeCount', 'loveCount', 'sadCount',
       'thankfulCount', 'wowCount', 'commentCount',
       'shareCount']
    cols_pred = [f"pred_{y_col}" for y_col in reactions]
    cols_gt = [f"actual_{y_col}" for y_col in reactions]

    X_test = pd.read_csv(f'{output_folder}/X_test.csv', index_col='id')

    predictions = pd.DataFrame(test_predictions, columns = cols_pred)
    ys_test = pd.DataFrame(test_labels, columns = cols_gt)
    predictions['id'] = X_test.index
    predictions.set_index('id', inplace=True)

    output = pd.concat([X_test, predictions], axis = 1)
    output.reset_index(inplace=True)
    ys_test.reset_index(inplace=True)
    output = pd.concat([output,  ys_test], axis = 1)
    output.to_csv(f'{output_folder}/test_predictions-LR-{LR}.csv')


    results = {}
    for y_col in reactions:
        y_col_true = 'actual_' + y_col
        y_col_pred = 'pred_' + y_col
        mse = mean_squared_error(output[y_col_true], output[y_col_pred])
        mae = mean_absolute_error(output[y_col_true], output[y_col_pred])
        medae = median_absolute_error(output[y_col_true], output[y_col_pred])
        mape = mean_absolute_percentage_error(output[y_col_true], output[y_col_pred])
        results[y_col] = {
            'mse': float(mse),
            'mae': float(mae),
            'medae': float(medae),
            'mape': float(mape)
        }


    with open(f'{output_folder}/test_scores-LR-{LR}.json', 'w') as f:
        json.dump(results, f)
