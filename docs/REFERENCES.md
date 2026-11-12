# References

Literature behind the solution in `solution/`, and how each source maps onto our rules. Collected on
2026-11-12.

Reading status:
- **Read**: the methods (and results) were read from the source itself.
- **Abstract**: only the abstract and bibliographic record could be read.
- **Not reachable**: the full text was behind a login, blocked automated access, or was a scanned
  PDF without text. Nothing beyond the bibliographic record is used from these.

## 1. Possession detection from tracking data

### Vidal-Codina, Evans, El Fakir & Billingham (2022). Read (main text).
*Automatic event detection in football using tracking data.* Sports Engineering 25, 18.
DOI [10.1007/s12283-022-00381-6](https://link.springer.com/doi/10.1007/s12283-022-00381-6).
Open-access versions: [arXiv 2202.00804](https://arxiv.org/abs/2202.00804),
[MIT DSpace](https://dspace.mit.edu/handle/1721.1/145347). MIT Sports Lab with FIFA.

- Possession zone: a player possesses the ball when it is inside a circle of radius 0.5 m
  (Track160) or 1 m (Tracab, Hawk-Eye) around him (Sec. 2.3.1, 3).
- Duel zone: a circle of 1 m around the ball; two or more opponents inside it make the frame a
  duel, and a duel overrides possession (Sec. 2.3.1).
- A gain of possession needs a change of ball direction or speed during the control frames;
  control runs without such a change are discarded as false positives (Sec. 2.3.3).
- Noisy ball data is smoothed with a Savitzky-Golay filter, 2nd order, 7 frames at 25 Hz
  (Sec. 2.3.1).
- Dead ball comes from the provider's in-play flag or from missing ball data; set pieces from
  player-position rules (Sec. 2.3.4). Passes detected with more than 90% precision and recall;
  corners, throw-ins and goal kicks above 90%, free kicks weakest (Sec. 3.1).
- Player possession itself is not benchmarked (Sec. 3.1). Main stated limitation: ball tracking
  data must be available (Sec. 4).
- The numeric thresholds of the possession loss/gain tests are in the supplement (Online
  Resource 1, Sec. S3). **Not reachable:** Springer requires a login and the MIT DSpace download
  was rate-limited.

**Used for:** the 1 m control radius (`R_CONTROL`), contested balls give no credit (`MARGIN`), a
touch is a change of ball velocity. **Next:** a duel zone (both teams within 1 m = no evidence)
and Savitzky-Golay smoothing before computing ball acceleration.

### Link & Hoernig (2017). Read (Sections 1-5.1).
*Individual ball possession in soccer.* PLOS ONE 12(7): e0179953.
DOI [10.1371/journal.pone.0179953](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/).

- Only in-play time is assigned to a player or team (Sec. 2).
- Individual possession begins "the moment a player is able to perform an action with the ball"
  and ends when another player's begins. Team possession begins with the first individual
  possession of one of its players and ends with the first of an opponent (Sec. 2). Headed or
  "ping-pong" balls are possession, though not control.
- Possession candidates: the nearest player within a reach threshold. Kick detection adds a local
  maximum of the smoothed ball acceleration of at least 4 m/s² (Sec. 3.2). Thresholds other than
  the acceleration were set by grid search and are not reported.
- Smoothing: trajectory jumps removed, segments under 1 s dropped, Rauch-Tung-Striebel smoother.
- Ball prediction 1 s ahead to decide when a player can no longer act on the ball (Sec. 3.3).
- Data: 60 Bundesliga matches, TRACAB, 25 Hz. Team ball possession 56:04 ± 5:12 min per match.
- Kick detection F = .83 (precision .86, recall .80) against video labels, ±0.6 s tolerance
  (Table 1). Inter-observer agreement of the human labels κ = .72.

**Used for:** the definition we follow (a single touch, even a header, starts a team possession;
this matches the public ground truth). **Next:** replace our impulse test (3 m/s velocity change
within 0.12 s, about 25 m/s²) with the local-maximum acceleration test at 4 m/s².

### Hoernig, Link, Herrmann, Radig & Lames (2016). Not reachable.
*Detection of individual ball possession in soccer.* Proc. 10th Int. Symposium on Computer
Science in Sports, Springer AISC 392, pp. 103-107. DOI 10.1007/978-3-319-24560-7_13.
Conference version of the method above; Link & Hoernig (2017) describe it and are cited instead.

### Kim, Lee, Seo, Ko, Yoon & Park (2026). Read (abstract, method, results summary).
*PathCRF: Ball-free soccer event detection via possession path inference from player
trajectories.* KDD 2026. [arXiv 2602.12080](https://arxiv.org/abs/2602.12080).
Code: [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf.git).

- Infers who controls the ball, and passes between players, from player trajectories only: a
  conditional random field over a graph of 22 players plus 4 "outside" nodes, decoded with
  Viterbi; only physically possible transitions are allowed.
- Supervised: trained on event labels of public Bundesliga matches (DFL/Sportec). Uses in-play
  segments only (stoppages excluded), 10 s windows, 5 fps.
- Frame-level team possession accuracy 92.4%; event F1 75.7%. Accuracy drops across tracking
  providers.

**Relevance:** confirms that team possession can be inferred from player movement alone, which
is our situation for the 41% of graded frames without a ball. Not usable as is: it needs a model
trained on labelled matches of another provider, and it does not label dead ball. Its use of
Viterbi with forbidden transitions supports our v4 design.

### Khaustov & Mozgovoy (2020). Not reachable.
*Recognizing events in spatiotemporal soccer data.* Applied Sciences 10(22), 8046.
[mdpi.com/2076-3417/10/22/8046](https://mdpi.com/2076-3417/10/22/8046) (refused automated access).
A rule-based detector for possession, passes and shots, proposed as a simple baseline.

## 2. State models

### Rabiner (1989). Not reachable (scanned PDF without text).
*A tutorial on hidden Markov models and selected applications in speech recognition.*
Proceedings of the IEEE 77(2), 257-286. DOI 10.1109/5.18626.
The standard reference for HMMs and the Viterbi algorithm used in `possession.py` (v4). The
property we rely on is textbook: a state with self-transition probability a stays for a
geometric number of steps with mean 1 / (1 - a). This was not checked against the paper's text.

### López Peña (2014). Read.
*A Markovian model for association football possession and its outcomes.* Preprint,
[arXiv 1403.7993](https://arxiv.org/abs/1403.7993).

- Possession as a Markov chain over game events with constant keep / loss / shot probabilities,
  fitted to all 380 Premier League 2012-13 matches (Opta).
- The possession length is therefore geometric; fitted keep probability 0.749-0.797 per event
  across teams (mean length about 4-5 events). An extra "accumulation" term improves the fit for
  long possessions.

**Used for:** a constant per-frame change probability (our `SWITCH_COST`) is the geometric model
this paper fits. It gives no per-frame rate, so the value itself still comes from durations.

### Ötting & Karlis (2023). Abstract.
*Football tracking data: a copula-based hidden Markov model for classification of tactics in
football.* Annals of Operations Research 325(1), 167-183. DOI 10.1007/s10479-022-04660-0.
An HMM on football tracking data whose hidden states are tactics (observed: each team's effective
playing space). Precedent for HMMs on this kind of data; it says nothing about possession.

## 3. Numbers we rely on

| Our value | Literature | Status |
|---|---|---|
| Dead ball 42.4% of graded frames (v3, v4) | CIES Football Observatory Monthly Report 64 (Poli, Ravenel & Besson, 2021; 37 competitions, 2019-21, InStat): ball in play 61.3% on average, **LaLiga 59.3%**, average match 96'14" ([report](https://football-observatory.com/IMG/sites/mr/mr64/en/)). Linke, Link, Weber & Lames (2018), JSSM 17(4), 662-667: effective time 66.3% of the first 15 min falling to 55.9% of the last 15 min ([full text](https://www.jssm.org/jssm-17-662.xml-Fulltext)); they cite Siegle & Lames (2012), J. Sports Sci. 30, 619-624, for "matches are halted on average for 38% of the total match time". FIFA U-20 World Cup 2025: ball in play 54:01 per match ([FIFA Training Centre](https://www.fifatrainingcentre.com/en/game/tournaments/fu20wc/2025/ball-in-play-time-and-possession-phases.php)) | **Consistent:** about 38-45% dead is expected, LaLiga ~41% |
| `SLOW_PLAYERS` 1.4 m/s | Bohannon (1997), Age and Ageing 26(1), 15-19, DOI [10.1093/ageing/26.1.15](https://academic.oup.com/ageing/article-lookup/doi/10.1093/ageing/26.1.15): 230 healthy adults, mean comfortable walking speed 1.27 m/s (women, 70s) to 1.46 m/s (men, 40s) | Supported |
| `PASS_SPEED` 6 m/s, `RESTART_SPEED` 5 m/s | Pfaff et al. (2022), *Controlling ball progression in soccer*, [arXiv 2210.16474](https://arxiv.org/abs/2210.16474): initial pass speed 16.20 m/s, deceleration 7.93 m/s² (Allsvenskan 2021, Signality, 25 fps). A pass therefore stays above 6 m/s for about 1.3 s | Consistent as a lower bound |
| `R_CONTROL` 1 m | Possession zone 0.5-1 m (Vidal-Codina et al. 2022) | Supported |
| `SWITCH_COST` = ln(375), from a 15 s mean possession | FIFA U-20 World Cup 2025: mean possession sequence 8.1 s (2023) and 8.8 s (2025); the page gives no definition of a sequence. Link & Hoernig (2017) and López Peña (2014) give no team possession duration in seconds | **Weak:** 8.8 s would give ln(220) = 5.4. Definitions differ (our ground truth: 17 possessions of mean 14.2 s in the public window, but that window must not be used to tune). Revisit with a reason |
| Contested ball keeps the last team | Vidal-Codina et al. (2022) duel zone; FIFA U-20 2025 reports the ball "in contest" 6% of the time; the challenge rules say a loose or contested ball stays with the last team | Supported |
| `EVIDENCE_RADIUS` / `R_TOUCH` 2 m | No source; chosen for the noisier automatic ATD | Assumption |
| `EVIDENCE_SCALE` 1 m, `EVIDENCE_CAP` 0.5 | No source | Assumption |
| `IMPULSE` 3 m/s within 0.12 s (about 25 m/s²) | Kick = local maximum of acceleration >= 4 m/s² on smoothed data (Link & Hoernig 2017) | **Contradicted:** explains why our touch detector rarely fires |

## 4. What happened when we used the literature values as they are (v5)

v5 took the structure and the values from the papers: Savitzky-Golay smoothing (7 frames, 2nd
order), kick = acceleration peak >= 4 m/s² with a player within 1 m, a 1 m possession zone and a
1 m duel zone (all from Link & Hoernig 2017 and Vidal-Codina et al. 2022), and a 12 s mean
possession (FIFA U-20 2025, Spain).

On our automatic ATD the values did not transfer:

- The median smoothed ball acceleration over all ball-visible frames is already 5.7 m/s² (p90
  18.1, p99 36.4). The 4 m/s² threshold was set on cleaned TRACAB data; here it sits below the
  noise floor and no longer separates kicks from rolling.
- The ball is within 1 m of the nearest player on only 28.5% of ball-visible frames (median
  1.9 m, explore.py), so a 1 m zone gives too little evidence and real team changes are missed.

Result: 216 possessions of mean 16.9 s on the full match (v4: 262 of 14.0 s); on the public window
9 possessions instead of 17 and 1 of 12 live team changes caught (v4: 6).

Vidal-Codina et al. (2022) already use a different possession radius per tracking provider and
recommend tuning per provider. v6 keeps the literature's structure and calibrates the values on
the distribution of our own match data, without labels.
