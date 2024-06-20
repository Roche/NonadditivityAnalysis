# Classification submodule of the NAA package

use --classify in the commandline to trigger the slgorithms contained here.

Classification is done in a bottom up approach, first classifying the Compounds present in NA Circles, then the transforamtions and last the Circles.

- [classification_classes](classification_classes) contain the classes needed in the [classify.py](classify.py) workflow.
- *_classification.py contain the code that is called in the [classification_classes](classification_classes) for every class respectively that works with the rdkit functionlaity.
- [ortho_classification.py](ortho_classification.py) contains functionality for the ortho classification across all three levels of the [classification_classes](classification_classes/)
- [rgroup_distance.py](rgroup_distance.py) contains functionality for the calculation of the rgroup distance over all three levels of[classification_classes](classification_classes/)
- [utils.py](utils.py) helper functions.
- [classify.py](classify.py) contains a wrapper function that is called in the [nonadditivity_workflow](../nonadditivity_workflow.py).
