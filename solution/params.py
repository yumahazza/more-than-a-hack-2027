"""Every threshold of the solution, in one place, with the reason for its value.

Rule we follow (RULES.md / EVALUATION.md): values come from football logic, from the literature
(docs/REFERENCES.md) and from aggregate statistics of the match data (solution/explore*.py), NEVER
from trying values against the public 5-minute ground truth. Change a value only with a reason you
could write in the technical document.
"""

FPS = 25

# -- kinematics ----------------------------------------------------------------------------------
# Ball velocity "before" and "after" a frame is measured over K_VEL frames (0.12 s): long enough to
# average out position noise, short enough that a touch is not smeared over a whole pass.
K_VEL = 3
# Player activity: speed per track slot over a 0.4 s span, slots faster than a sprint are identity
# switches and are dropped, then the mean over players is smoothed over +/- 1 s.
ACTIVITY_SPAN = 5          # frames each side -> 10-frame (0.4 s) displacement
MAX_PLAYER_SPEED = 12.0    # m/s; a top sprint is ~10 m/s
ACTIVITY_SMOOTH = FPS      # +/- 1 s

# -- kicks (v5, from the literature) -------------------------------------------------------------
# Link & Hoernig (2017): a kick is a local maximum of the SMOOTHED ball acceleration of at least
# 4 m/s^2, credited to the nearest player within reach. Smoothing as in Vidal-Codina et al. (2022):
# Savitzky-Golay, 2nd order, 7 frames at 25 Hz; the derivatives come straight from the filter.
# This replaces the v1-v4 impulse test (3 m/s within 0.12 s, i.e. ~25 m/s^2), which diagnose.py
# showed almost never fires on the smooth ATD ball.
SG_WINDOW = 7              # frames
KICK_ACCEL = 4.0           # m/s^2
# Possession zone and duel zone, Vidal-Codina et al. (2022): a player possesses the ball when it is
# within 1 m of him (Tracab and Hawk-Eye; 0.5 m for a third provider); when players of BOTH teams
# are within 1 m of the ball it is a duel, which overrides possession. The challenge rules agree:
# a contested ball stays with the last team, so a duel gives no evidence and credits no kick.
R_POSSESSION = 1.0         # m
R_DUEL = 1.0               # m

# -- touches (v1-v3 only; kept for assign_events and diagnose.py) --------------------------------
# A touch is a sudden change of ball velocity with a player close to it. A pass or a deflection
# changes the velocity by several m/s; tracking noise on the computed speed is well under 1 m/s.
IMPULSE = 3.0              # m/s change of ball velocity
# v2: 2.0 m (was 1.5). A tracked player is the centre of the body; a stretched leg plus the 2-D
# projection error of a ball off the ground put the ball up to ~2 m from it at the touch.
R_TOUCH = 2.0              # m from the ball
# Dribbling: a slow ball staying close to one player is controlled by him even without an impulse.
R_CONTROL = 1.0            # m
CONTROL_SPEED = 4.0        # m/s; faster than a jog-dribble means the ball is travelling
# v2: 0.6 s and a clear 1 m lead (was 0.2 s / 0.4 m). A defender standing next to a slow ball for a
# moment is not dribbling it; a player in control keeps it close over several touches.
CONTROL_MIN = 15           # frames (0.6 s) of continuous control
CONTROL_MARGIN = 1.0       # m closer than the nearest opponent
# If both teams are about equally close, the ball is contested: the rules say possession stays with
# the last team, so no touch is credited.
MARGIN = 0.4               # m

# -- dead ball -----------------------------------------------------------------------------------
# explore.py: median mean-player speed is ~2 m/s with the ball in view and ~1 m/s while it is lost
# (mostly stoppages). Walking pace is ~1.2 m/s.
# v2: 1.4 m/s (was 1.2), the usual human walking speed. In a stoppage (foul, injury) players walk.
SLOW_PLAYERS = 1.4         # m/s, smoothed mean player speed
# v2: the ball does not have to be still in a stoppage (it is carried, rolled to the taker). What
# a stoppage never has is a pass: no ball faster than a short pass within +/- 1 s.
PASS_SPEED = 6.0           # m/s; a short pass travels at 6-10 m/s
# v3: a pass (or a restart kick) is fast for a SUSTAINED stretch, not one frame: a ball kicked
# 6-10 m/s keeps that speed for well over 0.2 s. Single-frame spikes are tracking noise; v2 used a
# plain max and every spike broke a stoppage or faked a restart.
FAST_MIN = 5               # frames (0.2 s)
QUIET_WINDOW = FPS         # +/- 1 s
# v3: a 5 s walk without a pass is a stoppage only where Smart Tagging does NOT describe a phase of
# play (explore2: lost-ball stretches covered by phase tags are mostly live play).
LIVE_TAG_MAX = 0.5         # share of the stretch covered by phase-of-play tags
STOP_MIN = 3 * FPS         # a stoppage shorter than 3 s is not worth a D (restarts take longer)
STOP_ALONE = 5 * FPS       # walking players and no pass for 5 s is a stoppage even without a tag
STOP_LONG = 8 * FPS        # a long quiet stretch with the ball lost is a stoppage even untagged
STOP_EXTEND = 15 * FPS     # a stoppage lasts until the restart, looked for up to 15 s later
STOP_EVIDENCE = 10 * FPS   # a set piece tag starting within +/- 10 s supports a stoppage
CLOSE_HOLES = FPS          # holes <= 1 s inside a quiet stretch are closed
# Ball out of play: explore.py found 81 out-of-bounds runs, median 21 frames.
OOB_MARGIN = 0.3           # m beyond the line, to stay clear of noise at the line
OOB_MIN = 3                # frames
# v2: a restart is a KICK OR THROW OF A BALL THAT WAS STILL (or held, i.e. lost by the tracker).
# v1 took any ball moving at 3 m/s back inside the pitch, which fired while the ball was being
# carried back for a goal kick.
RESTART_SPEED = 5.0        # m/s
RESTART_STILL = 1.5        # m/s; the ball before the restart is placed or held ...
RESTART_STILL_SPAN = FPS // 2  # ... for at least 0.5 s
RESTART_INSIDE = 0.5       # m inside the pitch
RESTART_MIN_DELAY = 2 * FPS  # no restart can follow a ball going out in under 2 s
RESTART_MAX = 60 * FPS     # give up looking after 60 s
OOB_FALLBACK = 10 * FPS    # no restart found -> 10 s of dead ball
KICKOFF_RADIUS = 4.0       # m from the centre spot
KICKOFF_SEARCH = 300 * FPS
MIN_DEAD = FPS             # a dead stretch under 1 s is noise
MERGE_DEAD = 2 * FPS       # two dead stretches less than 2 s apart are one stoppage

# -- possession ----------------------------------------------------------------------------------
# v4: team in live play is the most likely H/A sequence of a two-state hidden Markov model,
# decoded with Viterbi, instead of switching on single touch events.
# Why: diagnose.py showed the velocity-change ("impulse") touch detector almost never fires at a
# real change: the ATD ball path is smooth (explore2: speed over 1 frame and over 5 frames have the
# same distribution), so a touch bends the path over many frames instead of in one.
# Evidence per frame (log-likelihood ratio, + = home):
#   v5: EVIDENCE_CAP for the team whose player has the ball in his possession zone (1 m, see above),
#   zero in a duel or when nobody is within 1 m (ball in flight). v4 used a 2 m radius and a graded
#   distance lead, two values without a source; the literature zone replaces both.
EVIDENCE_CAP = 0.5         # per frame; one frame can never decide alone (assumption, no source)
TAG_EVIDENCE = 0.05        # per frame while the ball is lost and a reliable tag names a team
# Cost of a team change = log of the expected frames between changes; a constant cost per frame
# is the geometric possession length that Markov models of possession fit (Lopez Pena 2014).
# v5: mean possession 12 s (was 15 s, no source). FIFA U-20 World Cup 2025 reports possession
# sequences of 8.8 s on average and 12.0 s for Spain, the most possession-oriented sides being
# 11.8-12.5 s. This is a Spanish top-flight match with a possession-oriented home side.
# 12 s = 300 frames -> ln(300) = 5.70. A clear zone lead needs ~0.5 s of play to pay for a change.
MEAN_POSSESSION_S = 12.0
SWITCH_COST = 5.70         # = ln(MEAN_POSSESSION_S * FPS)
# A kick by one team, not in a duel, is one contact, and one contact starts a team possession
# (Link & Hoernig 2017; also the public ground truth's own definition). It is worth one change.
KICK_EVIDENCE = SWITCH_COST
START_PRIOR = 20.0         # the restart team is near certain at the first frame of live play
START_LOOKAHEAD = 5 * FPS  # after a restart, the first kick within 5 s names the team
SETPIECE_WINDOW = 5 * FPS  # a set-piece tag whose window +/- 5 s covers the restart names it too
GAP_TAG_MIN = 2 * FPS      # while the ball is lost, a tag must point to the other team for 2 s
MIN_POSSESSION = 5         # frames; shorter runs are absorbed into the previous possession

# Smart Tagging reliability (explore.py: agreement with nearest player, 76% build up ... 54% box)
TAG_WEIGHTS = {
    "build up": 1.0, "defending in attacking third": 1.0,
    "progression": 1.0, "defending in middle third": 1.0,
    "attacking transition": 0.8, "defensive transition": 0.8,
    "long ball": 0.6, "ball in final third": 0.6,
    "creating chances": 0.5, "defending in defensive third": 0.5,
    "ball in the box": 0.2, "players in the box": 0.2,
}
TAG_MARGIN = 0.5           # weighted vote lead needed to name a team
