import os
import sendgrid
from sendgrid.helpers.mail import *


class MailSender(object):
    """
    A wrapper class for sending email using SendGrid's API
    Usage:

    >>> from utils.email import MailSender
    >>> email = MailSender()
    >>> email.send("abc@gmail.com", "Hello World", "Hello!")
    >>> ...

    """

    def __init__(self, config_path='./'):
        self.sender = Email(os.environ.get('email_addr'))
        self.api_key = os.environ.get('email_api_key')
        self.sg = sendgrid.SendGridAPIClient(apikey=self.api_key)

    def send(self, target, subject, text=None, html=None):
        if isinstance(target, list):
            target = list(set(target))
            for tg in target:
                try:
                    self._send(tg, subject, text, html)
                except:
                    pass
        else:
            self._send(target, subject, text, html)

    def _send(self, target, subject, text=None, html=None):
        receiver = Email(target)
        if text is not None:
            content = Content("text/plain", text)
        elif html is not None:
            content = Content("text/html", html)
        else:
            raise ValueError('Email has no body')
        mail = Mail(self.sender, subject, receiver, content)
        response = self.sg.client.mail.send.post(request_body=mail.get())
        return response
