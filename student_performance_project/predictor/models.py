from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):
    """Stores every prediction a user makes."""
    user             = models.ForeignKey(User, on_delete=models.CASCADE)
    study_hours      = models.FloatField()
    sleep_hours      = models.FloatField()
    social_media     = models.FloatField()
    screen_time      = models.FloatField()
    diet_quality     = models.IntegerField()
    mental_health    = models.IntegerField()
    physical_activity= models.FloatField()
    attendance       = models.FloatField()
    stress_level     = models.IntegerField()
    internet_usage   = models.FloatField()
    extra_curricular = models.IntegerField()
    time_management  = models.FloatField()
    predicted_label  = models.CharField(max_length=20)
    best_model       = models.CharField(max_length=50)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} → {self.predicted_label} ({self.created_at:%Y-%m-%d})"
