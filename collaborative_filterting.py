
# pip3 install numpy
# pip3 install scipy
import sys
import warnings

import os
import numpy as np
import itertools

from typing import List
from dataclasses import dataclass
from scipy.stats import pearsonr 


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def create_train_eval_dataset(fs_path: str):
    """
    Create the trainig and eval dataset from original data set, we use 10%
    of the data for evaluation and 90% for training.
    """

    print("Creating training and evaluation dataset......................................")
    training_set = list()
    evaluation_set = list()

    file_name, file_extention =  os.path.splitext(fs_path)

    # Reading the dateset input file.
    with open(fs_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            if index % 10 == 0:
                evaluation_set.append(line)
            else:
                training_set.append(line)

    # Writing the evaluation file in same format.
    with open(file_name + ".eval", "w") as f:
        for line in evaluation_set:
            f.write(line)
    
    # Writing the training file in same format.
    with open(file_name + ".train", "w") as f:
        for line in training_set:
            f.write(line)


def evaluate_model(predict_fs_path: str, eval_fs_path: str):
    """
    Run evaluation on on original data and predicted data and caculate root mean 
    square error at the end. 
    """
    print("Evaluating the model, please wait, this can take few mins.....................")

    # Read the predicted file and build cache for lookup for each prediced user and item.
    predict_data_cache = dict()
    with open(predict_fs_path, "r") as f:
       for line in f.readlines():
           values = [int(s) for s in line.replace("\n", "").split(" ")]
           key = str(values[0])+ "_" + str(values[1])
           predict_data_cache[key] = values[2]

    # Read eval file with original rating and calculate the difference square.
    evals = list()
    with open(eval_fs_path, "r") as f:
        for line in f.readlines():
            values = [int(s) for s in line.replace("\n", "").split(" ")]
            key = str(values[0])+ "_" + str(values[1])
            predict_rating = predict_data_cache.get(key)
            original_rating = values[2]
            evals.append((original_rating - predict_rating)**2)
    
    # Calculate RMSE
    print("Root Mean Square Error is :", np.sqrt(np.mean(evals)))

def train_model(input_file: str, total_users:int, total_items:int, total_neighours:int, output_file:str):
    """
    We use collaborating filtering for training, for a given user we calculate similar user and 
    for a given item we calculate similar items. When we need to predict rating we would than use 
    similar user weighted average to predict rating, if we cannot predict rating than we would use
    similar items weighted average to predict rating, if we still cannot predict the rating its a 
    cold start problem, we default to average rating(3).

    This function at the end returns a matrix that has all the predicted values populated for a 
    user and item
    """
    print("Training the model, please wait, this can take few mins.......................")
    dataset = _load_dataset(input_file, total_users, total_items)
    rating_matrix = _create_matrix(total_users, total_items, dataset)

    # Calcuate neighbours for users and building the cache of user with neighbor.
    user_neighbors_cache = dict()
    for user_id in range(1, total_users + 1):
        # print(f"Calculating neighbours for user id {user_id}")
        neigbours_ids_for_user = _calculate_id_neighbours(rating_matrix, total_users, total_neighours, user_id, False)
        user_neighbors_cache[user_id] = neigbours_ids_for_user

    # Calculate neighbours for items and build the cache of items with neighbor.
    item_neighbors_cache = dict()
    for item_id in range(1, total_items + 1):
        # print(f"Calculating neighbours for item id {item_id}")
        neigbours_ids_for_item = _calculate_id_neighbours(rating_matrix, total_items, total_neighours, item_id, True)
        item_neighbors_cache[item_id] = neigbours_ids_for_item

    final_matrix = np.zeros((total_users, total_items))

    total_user_base_predictions = 0
    total_item_base_predictions = 0 
    cold_start_predictions = 0
    given_ratings = 0

    for user_id in range(1, total_users + 1):
        for item_id in range(1, total_items + 1):
            if rating_matrix[user_id-1, item_id-1] == 0:
                # First prediction based on item similarities
                item_neighbors = item_neighbors_cache[item_id]
                predicted_item_based = int(np.average([(rating_matrix[user_id -1 , n - 1]) for n in item_neighbors]))
                # No similarities found, we fallback to user based prediction
                if predicted_item_based == 0:
                    user_neighbors = user_neighbors_cache[user_id]
                    rated_used_neighbours = [n for n in user_neighbors if rating_matrix[n -1, item_id - 1] != 0]
                    # Found user base prediction 
                    if rated_used_neighbours:
                        predicted_user_based = int(np.average([rating_matrix[n -1, item_id - 1] for n in rated_used_neighbours]))
                        final_matrix[user_id -1, item_id-1] = predicted_user_based
                        total_user_base_predictions = total_user_base_predictions + 1
                    else:
                        # Cannot find any item base or user based prediction, so we give average rating of 3
                        final_matrix[user_id -1, item_id-1] = 3
                        cold_start_predictions = cold_start_predictions + 1
                else:
                    final_matrix[user_id -1, item_id-1] = predicted_item_based
                    total_item_base_predictions = total_item_base_predictions + 1
            else:
                original_rating = rating_matrix[user_id-1, item_id-1]
                final_matrix[user_id -1, item_id-1] = original_rating
                given_ratings = given_ratings + 1
            #print(f"User:{user_id} Item:{item_id} Rating:{final_matrix[user_id -1, item_id-1]}")
    
    #print(f"total_user_base_predictions : {total_user_base_predictions}")
    #print(f"total_item_base_predictions : {total_item_base_predictions}")
    #print(f"cold_start_predictions : {cold_start_predictions}")
    #print(f"given_ratings : {given_ratings}")
    
    _dump_matrix(output_file, final_matrix, total_users, total_items)

def _calculate_id_neighbours(rating_matrix:np.ndarray, total_ids:int, total_neighours:int, id: int, item_based:bool)-> List[int]:
    """
    we use the notion of centered cosine similarity to find the set of N of users which is the neighborhood and this neighborhood 
    consists of K users which are most similar to user i who have also rated item j. Once we have this info we can make predictions 
    for User i and item j. The simplest prediction is to just take the average ratings for the neighbourhood.
    """
    other_ids_coeff = list()
    id_to_coeff_dict = dict()    
    coeff_to_ids_dict = dict()

    for other_id in range(1, total_ids + 1):
        if id != other_id:
            id_ratings = None
            other_id_ratings = None
            if item_based:
                # Column
                id_ratings = rating_matrix[:, id - 1]
                other_id_ratings = rating_matrix[:, other_id - 1]
            else:
                # Rows
                id_ratings = rating_matrix[id - 1]
                other_id_ratings = rating_matrix[other_id - 1]
            coeff, _ = pearsonr(id_ratings, other_id_ratings)

            other_ids_coeff.append(coeff)
            id_to_coeff_dict[other_id] = coeff
            if str(coeff) not in coeff_to_ids_dict:
                coeff_to_ids_dict[str(coeff)] = [other_id]
            else:
                coeff_to_ids_dict[str(coeff)].append(other_id)
            
    # sort in descending 
    other_ids_coeff = sorted(other_ids_coeff, reverse=True)
    # find other user ids from coeff
    all_other_user_by_similarity_order = ([coeff_to_ids_dict.get(str(coeff)) for coeff in other_ids_coeff])
    # flatten the list
    all_other_user_by_similarity_order = list(itertools.chain(*all_other_user_by_similarity_order))
    # pick top n neighbours 
    neighours = all_other_user_by_similarity_order[: total_neighours]
    #print(id, [ (n, id_to_coeff_dict.get(n)) for n in neighours])
    return neighours


@dataclass
class Data:
    user: int 
    item: int
    rating: int


def _create_matrix(total_users: int, total_items: int, rating_dataset: List[Data]) -> np.ndarray:
    output_matrix = np.zeros((total_users, total_items))
    
    for data in rating_dataset:
        output_matrix[data.user, data.item] = data.rating

    return output_matrix


def _load_dataset(fs_path: str, total_users: int, total_items: int) -> List[Data]:
    output = list()
    with open(fs_path, "r") as f:
        for line in f.readlines():
            values = [int(s) for s in line.replace("\n", "").split(" ")]
            data = Data(user=values[0] - 1, item=values[1] - 1, rating=values[2])
            #if data.user <= total_users and data.item <= total_items:
            output.append(Data(user=values[0] - 1, item=values[1] - 1, rating=values[2]))
    return output

def _dump_matrix(fs_path:str, matrix: np.ndarray, total_users:int, total_items:int):
    with open(fs_path, "w") as f:
        for i in range(1, total_users + 1):
            for j in range(1, total_items + 1):
                f.write(f"{i} {j} {int(matrix[i-1][j-1])}\n")


if __name__ == "__main__":
    total_users = 943
    total_items = 1682
    dataset_file = "train.txt"
    train_file = "dataset.train"
    predict_file = "dataset.predict"
    eval_file = "dataset.eval"
    total_neighours = 10

    create_train_eval_dataset(dataset_file)
    train_model(train_file, total_users, total_items, total_neighours, predict_file)
    evaluate_model(predict_file, eval_file)