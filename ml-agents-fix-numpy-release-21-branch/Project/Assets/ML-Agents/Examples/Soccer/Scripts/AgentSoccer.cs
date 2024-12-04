using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

public enum Team
{
    Blue = 0,
    Purple = 1
}

public class AgentSoccer : Agent
{
    // Note that that the detectable tags are different for the blue and purple teams. The order is
    // * ball
    // * own goal
    // * opposing goal
    // * wall
    // * own teammate
    // * opposing player

    public enum Position
    {
        Striker,
        Goalie,
        Generic
    }

    [HideInInspector]
    public Team team;
    float m_KickPower;
    // The coefficient for the reward for colliding with a ball. Set using curriculum.
    float m_BallTouch;
    public Position position;

    const float k_Power = 2000f;
    float m_Existential;
    float m_LateralSpeed;
    float m_ForwardSpeed;


    [HideInInspector]
    public Rigidbody agentRb;
    SoccerSettings m_SoccerSettings;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public float rotSign;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        if (envController != null)
        {
            m_Existential = 1f / envController.MaxEnvironmentSteps;
        }
        else
        {
            m_Existential = 1f / MaxStep;
        }

        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        if (position == Position.Goalie)
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.0f;
        }
        else if (position == Position.Striker)
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.3f;
        }
        else
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.0f;
        }
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        m_KickPower = 0f;

        var forwardAxis = act[0];
        var rightAxis = act[1];
        var rotateAxis = act[2];

        switch (forwardAxis)
        {
            case 1:
                dirToGo = transform.forward * m_ForwardSpeed;
                m_KickPower = 1f;
                break;
            case 2:
                dirToGo = transform.forward * -m_ForwardSpeed;
                break;
        }

        switch (rightAxis)
        {
            case 1:
                dirToGo = transform.right * m_LateralSpeed;
                break;
            case 2:
                dirToGo = transform.right * -m_LateralSpeed;
                break;
        }

        switch (rotateAxis)
        {
            case 1:
                rotateDir = transform.up * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

   public override void OnActionReceived(ActionBuffers actionBuffers)
{
    if (position == Position.Goalie)
    {
        // Existential bonus for Goalies
        AddReward(m_Existential);
    }
    else if (position == Position.Striker)
    {
        // Existential penalty for Strikers
        AddReward(-m_Existential);
    }

    // Process discrete actions for movement
    MoveAgent(actionBuffers.DiscreteActions);

    // Vision rotation handling
    if (actionBuffers.DiscreteActions.Length > 3) // Check if vision action is defined
    {
        var visionRotate = actionBuffers.DiscreteActions[3];
        switch (visionRotate)
        {
            case 1: // Rotate vision left
                visionAngle -= 45f;
                break;
            case 2: // Rotate vision right
                visionAngle += 45f;
                break;
            default:
                break; // No vision change
        }

        // Clamp vision angle to 0-360 degrees
        if (visionAngle >= 360f) visionAngle -= 360f;
        if (visionAngle < 0f) visionAngle += 360f;
    }

    // Check vision
    CheckVision();
}


    public override void Heuristic(in ActionBuffers actionsOut)
{
    var discreteActionsOut = actionsOut.DiscreteActions;

    // Ensure enough actions are set for the configured branches
    discreteActionsOut[0] = Input.GetKey(KeyCode.W) ? 1 : (Input.GetKey(KeyCode.S) ? 2 : 0); // Forward/Backward
    discreteActionsOut[1] = Input.GetKey(KeyCode.E) ? 1 : (Input.GetKey(KeyCode.Q) ? 2 : 0); // Strafe
    discreteActionsOut[2] = Input.GetKey(KeyCode.A) ? 1 : (Input.GetKey(KeyCode.D) ? 2 : 0); // Rotate

    // Vision rotation
    if (discreteActionsOut.Length > 3) // Ensure we don’t exceed array bounds
    {
        discreteActionsOut[3] = Input.GetKey(KeyCode.LeftArrow) ? 1 : (Input.GetKey(KeyCode.RightArrow) ? 2 : 0); // Vision rotation
    }
}

    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = k_Power * m_KickPower;
        if (position == Position.Goalie)
        {
            force = k_Power;
        }
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.2f * m_BallTouch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
    }  


public float visionAngle; // Current vision direction (in degrees, relative to forward)
public float visionRange = 5f; // Maximum vision distance
public float visionFOV = 90f; // Field of view in degrees

/// <summary>
/// Simulates an independent vision cone, allowing the agent to look in a direction decoupled from its movement.
/// </summary>
public void CheckVision()
{
    // Calculate the forward direction of the vision cone
    visionAngle = Random.Range(0, 360); // Random angle between 0 and 360 degrees
    Quaternion visionRotation = Quaternion.Euler(0, visionAngle, 0);
    Vector3 visionDirection = visionRotation * transform.forward;

    // Detect objects within the vision range
    Collider[] objectsInRange = Physics.OverlapSphere(transform.position, visionRange);

    foreach (var obj in objectsInRange)
    {
        // Calculate the direction to the object
        Vector3 directionToObj = (obj.transform.position - transform.position).normalized;

        // Calculate the angle between the vision direction and the direction to the object
        float angleToObj = Vector3.Angle(visionDirection, directionToObj);

        // Check if the object is within the field of view
        if (angleToObj <= visionFOV / 2)
        {
            // Perform additional logic for detected objects
            Debug.Log($"Object in vision: {obj.name}");
        }
    }
}



}