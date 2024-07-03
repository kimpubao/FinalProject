from django import forms
from .models import Question

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question 
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 1}),
        }
