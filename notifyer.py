from notifypy import Notify
import time

# Function to send a notification. Takes in a custom message for the user
def send_notification(msg):
    
    notification = Notify()
    notification.title = "Postura incorrecta detectada!"
    notification.message = msg
    notification.icon = "npd_t.png"
    # Set the time the notification will be shown for to 3 seconds (3000 milliseconds)
    notification.timeout = 3000
    notification.send()


