
import models.lenght_detection as ld
import models.edit_distance as edit
import models.known_events as ke
import models.combine as com
import models.ngram as ngram
import models.ecvc as vc


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

_comb = {"Method": com.Combine, "Update": com.update_strategy}