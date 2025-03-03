from models._thresholds import Thresholds

import models.lenght_detection as ld
import models.edit_distance as edit
import models.known_events as ke
import models.combine as com
import models.ngram as ngram
import models.ecvc as vc


from typing import List


_list = {
    "KnowEvents": {
        "Method": ke.KnownEvents, "Update": ke.update_strategy,
    },
    "LengthDetection": {
        "Method": ld.LengthDetection, "Update": ld.update_strategy,
    },
    "ECVC": {
        "Method": vc.ECVC, "Update": vc.update_strategy,
    },
    "2-Gram": {
        "Method": ngram.NGram2, "Update": ngram.update_strategy,
    },
    "3-Gram": {
        "Method": ngram.NGram3, "Update": ngram.update_strategy,
    },
    "Edit": {
        "Method": edit.EditDistance, "Update": edit.update_strategy,
    }
}


def model_init(names: List[str], thres: Thresholds = Thresholds()) -> dict:
    """
    Return the model to be initialize and its updated_strategy

    Format: {'Method': Model, 'Update': update_strategy}
    """
    if len(names) == 1:
        name = names[0]
        return {
            "Method": _list[name]["Method"](thres=thres), "Update": _list[name]["Update"]
        }
    
    return {
        "Method": com.Combine(
            models=[_list[name]["Method"](thres=thres) for name in names], 
            update_funcs=[_list[name]["Update"] for name in names]
        ),
        "Update": com.update_strategy
    }
    

