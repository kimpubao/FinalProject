from django import forms
from .models import Answer, Question

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question 
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control'}),
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }