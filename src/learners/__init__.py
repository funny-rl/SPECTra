from typing import Union, Type, Dict
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .nq_learner_data_augmentation import NQLearnerDataAugmentation

REGISTRY:Dict[str, Union[Type[NQLearner], Type[DMAQ_qattenLearner], Type[NQLearnerDataAugmentation]]] = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner_data_augmentation"] = NQLearnerDataAugmentation
