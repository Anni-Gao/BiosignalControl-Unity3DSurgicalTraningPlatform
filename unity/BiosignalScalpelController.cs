// Control logic of bio-signals in Unity 3D (changed after integration)
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json.Linq;  // Requires to manual import Newtonsoft.Json package (version 3.2.1)

[RequireComponent(typeof(Transform))]
public class ScalpelController : MonoBehaviour
{
    private TrailRenderer trail;
    private bool btn1Held = false;      // Whether button 1 is pressed
    private bool wasEmitting = false;   // Whether ink was emitting in the previous frame

    // UDP data reception
    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile string latestScalpelState = null;

    void Start()
    {
        // Start UDP listening thread (port must match Python sender)
        udpClient = new UdpClient(5005); // Example: listening on port 5005
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        // Initialize TrailRenderer component
        trail = GetComponent<TrailRenderer>();
    }

    void OnDestroy()
    {
        // Close UDP client and thread
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Abort();
        if (udpClient != null)
            udpClient.Close();
    }

    void ReceiveData()
    {
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEP);
                string message = Encoding.UTF8.GetString(data);

                // Parse JSON and get "scalpel_state"
                JObject json = JObject.Parse(message);
                string state = json.Value<string>("scalpel_state");

                if (!string.IsNullOrEmpty(state))
                {
                    latestScalpelState = state;  // Update latest state; main thread will read it
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError("UDP receive error: " + ex.Message);
            }
        }
    }

    void Update()
    {
        // Control ink emission based on latest scalpel_state
        if (latestScalpelState == "move")
        {
            trail.emitting = true;
        }
        else if (latestScalpelState == "stop")
        {
            trail.emitting = false;
        }
    }
}
