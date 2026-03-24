def interpret_emotional_state(assessment):
    """Map M(S) metrics to expected emotional signatures"""
    
    m_s = assessment.m_s_score
    velocity = assessment.m_s_velocity
    
    emotional_prediction = {}
    
    # Current state
    if m_s > 5:
        emotional_prediction["baseline"] = "JOY/CONTENTMENT"
    elif m_s > 1:
        emotional_prediction["baseline"] = "CALM/STABLE"
    elif m_s > 0:
        emotional_prediction["baseline"] = "ANXIETY/STRESS"
    else:
        emotional_prediction["baseline"] = "FEAR/GRIEF"
    
    # Trajectory
    if velocity and velocity > 0.1:
        emotional_prediction["trajectory"] = "HOPE/EXCITEMENT"
    elif velocity and velocity < -0.1:
        emotional_prediction["trajectory"] = "DREAD/SADNESS"
    
    # Reality fracture
    if assessment.reality_fracture_detected:
        emotional_prediction["cognitive"] = "CONFUSION/UPSET"
    
    # Wisdom crystallization
    if assessment.wisdom_crystallization:
        emotional_prediction["insight"] = "JOY/RELIEF"
    
    return emotional_prediction
