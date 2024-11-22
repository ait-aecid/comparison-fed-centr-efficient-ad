
import models.lenght_detection as ld
import models.known_events as ke


_list = {
    "KnowEvents": {
        "Method": ke.KnownEvents, "Update": ke.update_strategy,
    },
    "LengthDetection": {
        "Method": ld.LengthDetection, "Update": ld.update_strategy,
    },
}