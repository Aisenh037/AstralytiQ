"""
📧 Notification Service
-----------------------
Handles email notifications using Resend API.
"""

import streamlit as st
import resend

class NotificationService:
    def __init__(self):
        self.api_key = st.secrets.get("RESEND_API_KEY")
        if self.api_key:
            resend.api_key = self.api_key
            self.enabled = True
        else:
            self.enabled = False
            
    def send_training_complete_email(self, user_email, model_name, metrics):
        """Send email when model training completes."""
        if not self.enabled:
            return False
            
        try:
            params = {
                "from": "AstralytiQ <onboarding@resend.dev>",
                "to": [user_email],
                "subject": f"🚀 Training Completed: {model_name}",
                "html": f"""
                <div style="font-family: sans-serif; padding: 20px;">
                    <h2>Model Training Success!</h2>
                    <p>Your model <strong>{model_name}</strong> has finished training.</p>
                    
                    <div style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
                        <h3>📊 Performance Metrics</h3>
                        <ul>
                            <li><strong>Accuracy:</strong> {metrics.get('accuracy', 'N/A')}</li>
                            <li><strong>MAE:</strong> {metrics.get('mae', 'N/A')}</li>
                            <li><strong>Status:</strong> Ready for Deployment</li>
                        </ul>
                    </div>
                    
                    <p>
                        <a href="https://astralytiq.streamlit.app" 
                           style="background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                           View Dashboard
                        </a>
                    </p>
                </div>
                """
            }
            
            email = resend.Emails.send(params)
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

def get_notification_service():
    return NotificationService()
