"""Every threshold of the v1 solution, in one place, with the reason for its value.

Rule we follow (RULES.md / EVALUATION.md): values come from football logic and from aggregate
statistics of the match data (solution/explore*.py), NEVER from trying values against the public
5-minute ground truth. Change a value only with a reason you could write in the technical document.
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

# -- touches (who plays the ball) ----------------------------------------------------------------
# A touch is a sudden change of ball velocity with a player close to it. A pass or a deflection
# changes the velocity by several m/s; tracking noise on the computed speed is well under 1 m/s.
IMPULSE = 3.0              # m/s change of ball velocity
R_TOUCH = 1.5              # m from the ball (foot reach + the 2-D projection error of a ball in air)
# Dribbling: a slow ball staying close to one player is controlled by him even without an impulse.
R_CONTROL = 1.0            # m
CONTROL_SPEED = 4.0        # m/s; faster than a jog-dribble means the ball is travelling
CONTROL_MIN = 5            # frames (0.2 s) of continuous control
# If both teams are about equally close, the ball is contested: the rules say possession stays with
# the last team, so no touch is credited.
MARGIN = 0.4               # m

# -- dead ball -----------------------------------------------------------------------------------
# explore.py: median mean-player speed is ~2 m/s with the ball in view and ~1 m/s while it is lost
# (mostly stoppages). Walking pace is ~1.2 m/s.
SLOW_PLAYERS = 1.2         # m/s, smoothed mean player speed
BALL_STILL = 1.0           # m/s; a ball placed for a restart
STOP_MIN = 3 * FPS         # a stoppage shorter than 3 s is not worth a D (restarts take longer)
STOP_LONG = 8 * FPS        # a long quiet stretch with the ball lost is a stoppage even untagged
STOP_EVIDENCE = 10 * FPS   # a set piece tag starting within +/- 10 s supports a stoppage
CLOSE_HOLES = FPS          # holes <= 1 s inside a quiet stretch are closed
# Ball out of play: explore.py found 81 out-of-bounds runs, median 21 frames.
OOB_MARGIN = 0.3           # m beyond the line, to stay clear of noise at the line
OOB_MIN = 3                # frames
RESTART_SPEED = 3.0        # m/s; a restart kick/throw
RESTART_INSIDE = 0.5       # m inside the pitch
RESTART_MIN_DELAY = 2 * FPS  # no restart can follow a ball going out in under 2 s
RESTART_MAX = 60 * FPS     # give up looking after 60 s
OOB_FALLBACK = 10 * FPS    # no restart found -> 10 s of dead ball
KICKOFF_RADIUS = 4.0       # m from the centre spot
KICKOFF_SEARCH = 300 * FPS
MIN_DEAD = FPS             # a dead stretch under 1 s is noise
MERGE_DEAD = 2 * FPS       # two dead stretches less than 2 s apart are one stoppage

# -- possession ----------------------------------------------------------------------------------
START_LOOKAHEAD = 5 * FPS  # after a restart, the first touch within 5 s names the team
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
