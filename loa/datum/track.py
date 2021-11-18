class TrackDatum(object):
    def __init__(self, observations, priors):
        for obs_type, obs in observations.items():
            self.ts = obs.ts
        self.observations = observations
        self.priors = priors
        self.latent_states = []

    def merge(self, other):
        for k, v in other.observations.items():
            self.observations[k] = v

    def score(self):
        cur_score = 0.
        for prior in self.priors:
            cur_score += prior.score(self.observations)
        return cur_score


class Track(object):
    def __init__(self):
        self.data = []
        self.is_finished = False

    def add_datum(self, frame, datum):
        self.data.append(datum)

    def finish(self):
        self.is_finished = True