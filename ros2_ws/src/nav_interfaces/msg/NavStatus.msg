# Published by planners for navigation server status monitoring.
string nav_state                # "idle", "planning", "navigating", "emergency_stopped", "recovering"
float64 distance_to_goal        # meters remaining to final goal
float64 distance_traveled       # meters traveled since goal was set
float64 elapsed_time            # seconds since goal was set
float64 current_x               # robot X in map frame
float64 current_y               # robot Y in map frame
bool goal_reached               # true when goal tolerance is satisfied
bool has_active_goal            # true when planner has a goal to work on
