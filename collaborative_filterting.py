
# pip3 install numpy
# pip3 install scipy
import numpy as np
import itertools

from typing import List
from dataclasses import dataclass
from scipy.stats import pearsonr 


def recommend_rating(rating_matrix:np.ndarray, total_users:int, total_neighours:int, user_id:int, item_id:int):
    neigbours_ids_for_user = _calculate_user_neighbours(rating_matrix, total_users, total_neighours, user_id)
    
    # Filter neighbours only which has rated before.
    rated_neighbours_ids = [neighbour_id for neighbour_id in neigbours_ids_for_user if rating_matrix[neighbour_id -1, item_id - 1] != 0]

    if not rated_neighbours_ids:
        return 0

    # average neighbour rating
    return(int(np.average([rating_matrix[neighbour_id -1, item_id - 1] for neighbour_id in rated_neighbours_ids])))


def _calculate_user_neighbours(rating_matrix:np.ndarray, total_users:int, total_neighours:int, user_id: int)-> List[int]:
    other_users_coeff = list()
    user_to_coeff_dict = dict()    
    coeff_to_users_dict = dict()

    for user_j in range(1, total_users + 1):
        if user_id != user_j:
            coeff = _pearson_correlation_coefficient(rating_matrix, user_id, user_j)
            
            other_users_coeff.append(coeff)

            user_to_coeff_dict[user_j] = coeff
            
            if str(coeff) not in coeff_to_users_dict:
                coeff_to_users_dict[str(coeff)] = [user_j]
            else:
                coeff_to_users_dict[str(coeff)].append(user_j)
            
    # sort in descending 
    other_users_coeff = sorted(other_users_coeff, reverse=True)

    # find other user ids from coeff
    all_other_user_by_similarity_order = ([coeff_to_users_dict.get(str(coeff)) for coeff in other_users_coeff])

    # flatten the list
    all_other_user_by_similarity_order = list(itertools.chain(*all_other_user_by_similarity_order))

    # pick top n neighbours 
    neighours = all_other_user_by_similarity_order[: total_neighours]

    print(user_id, [ (n, user_to_coeff_dict.get(n)) for n in neighours])
    
    return neighours

def _pearson_correlation_coefficient(rating_matrix:np.ndarray, user_first: int, user_second: int) -> float:
    """
    Assume user_a and user_b are index 1 user
    """
    user_first_ratings = rating_matrix[user_first - 1]
    user_second_ratings = rating_matrix[user_second - 1]
    correlation, _ = pearsonr(user_first_ratings, user_second_ratings)
    return correlation


@dataclass
class Data:
    user: int 
    item: int
    rating: int


def _create_matrix(total_users: int, total_items: int, rating_dataset: List[Data]) -> np.ndarray:
    """
    TODO.
    """
    output_matrix = np.zeros((total_users, total_items))
    
    for data in rating_dataset:
        output_matrix[data.user, data.item] = data.rating

    return output_matrix


def _load_dataset(fs_path: str) -> List[Data]:
    """
    TODO.
    """
    output = list()
    with open(fs_path, "r") as f:
        for line in f.readlines():
            values = [int(s) for s in line.replace("\n", "").split(" ")]
            output.append(Data(user=values[0] - 1, item=values[1] - 1, rating=values[2]))
    return output

if __name__ == "__main__":
    total_users = 4
    total_items = 7
    total_neighours = 2
    dataset = _load_dataset("/Users/nmurarka/Desktop/recommender_system/sample_test.txt")
    rating_matrix = _create_matrix(total_users, total_items, dataset)

    for i in range(1, total_users + 1):
        for j in range(1, total_items + 1):
            if rating_matrix[i-1, j-1] == 0:
                print(f"User:{i} Item:{j} Rating:{recommend_rating(rating_matrix, total_users, total_neighours, i, j)}")
        print("\n")
