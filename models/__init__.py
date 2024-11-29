
import models.lenght_detection as ld
import models.known_events as ke
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
}