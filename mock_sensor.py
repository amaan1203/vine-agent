import json
import time
import paho.mqtt.client as mqtt

broker = "127.0.0.1"
port = 1883
topic = "vine/sensors/BlockB"

# Simulate a live sensor payload
payload = {
    "block": "B", 
    "vwc_pct": 21.0, 
    "temp_f": 95.1,
    "co2_ppm": 415,
    "date": "2026-03-06",
    "time": "17:45"
}

print(f"Connecting to MQTT Broker (RabbitMQ) at {broker}:{port}...")
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(broker, port)
client.loop_start() # Start network loop to process packets

print(f"Publishing live sensor packet to {topic}...")
msg_info = client.publish(topic, json.dumps(payload), qos=1)
msg_info.wait_for_publish() # Wait for RabbitMQ to confirm receipt

client.loop_stop()
client.disconnect()

print("Payload verified and received by RabbitMQ! Now ask the Chatbot: 'Why are the vines in Block B struggling right now?'")
