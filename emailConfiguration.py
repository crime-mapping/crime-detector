# import os
# import smtplib
# from email.mime.text import MIMEText
# from dotenv import load_dotenv

# load_dotenv()

# def send_alert(camera_name, crime_type, severity):
#     user_email = "ishimweinstein@gmail.com"  # Fetch dynamically from DB in real scenario
#     subject = f"🚨 Crime Alert: {crime_type} detected!"
#     body = f"A crime of type **{crime_type}** was detected on **{camera_name}** with severity score **{severity}**."

#     msg = MIMEText(body)
#     msg["Subject"] = subject
#     msg["From"] = "noreply@smartsurveillancesystem.com"
#     msg["To"] = user_email

#     try:
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
#         server.login("ishimwe.nyanja@gmail.com", os.getenv("EMAIL_PASSWORD"))  
#         server.sendmail("noreply@smartsurveillancesystem.com", user_email, msg.as_string())
#         server.quit()
#         print(f"📧 Alert email sent to {user_email}")
#     except Exception as e:
#         print(f"❌ Email failed: {e}")


import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_alert(users,camera_name, crime_type, severity_score, severity_level):
    subject = f"🚨 Crime Alert: {crime_type} detected!"

    # Color mapping
    colors = {
        "LOW": "#FFF8E1",       # light yellow
        "MEDIUM": "#FFECB3",    # amber
        "HIGH": "#FFCDD2",      # light red
    }
    bg_color = colors.get(severity_level, "#FFFFFF")

    html_body = f"""
    <html>
    <body style="background-color: {bg_color}; font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #D32F2F;">🚨 Crime Detected!</h2>
        <p><strong>Type:</strong> {crime_type}</p>
        <p><strong>Location:</strong> {camera_name}</p>
        <p><strong>Severity Score:</strong> {round(severity_score, 3)} ({severity_level.title()})</p>
        <p>Please review the incident in your dashboard immediately.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = "noreply@smartsurveillancesystem.com"

    mime_text = MIMEText(html_body, "html")
    msg.attach(mime_text)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("ishimwe.nyanja@gmail.com", os.getenv("EMAIL_PASSWORD"))
        for user in users:
            if user.get("email"):
                msg["To"] = user["email"]
                server.sendmail(msg["From"], msg["To"], msg.as_string())
                print(f"📧 Alert email sent to {user['email']}")
        server.quit()
    except Exception as e:
        print(f"❌ Email failed: {e}")
