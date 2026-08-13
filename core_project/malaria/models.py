from django.db import models

# Create your models here.
# malaria/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import MinValueValidator

class SystemUser(AbstractUser):
    """
    Custom User Table: Restricts profile access rights.
    Ensures absolute alignment with the Zambia Data Protection Act (2021).
    """
    ROLE_CHOICES = (
        ('ADMIN', 'System Administrator'),
        ('EPIDEMIOLOGIST', 'Public Health Epidemiologist'),
        ('CLERK', 'Data Entry Clerk'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='CLERK')
    district_assignment = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

class ZambianDistrict(models.Model):
    """Stores target geographical nodes for matching weather strings."""
    name = models.CharField(max_length=100, unique=True)
    province = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    population = models.PositiveIntegerField(
        default=0,
        help_text="Estimated resident population, used to compute cases-per-10,000 incidence rate. "
                   "Leave as 0 if unknown — hotspot tiers fall back to raw case counts."
    )

    def __str__(self):
        return f"{self.name} District"

class IntegratedMalariaData(models.Model):
    """Unified Table: Cross-references NMEC clinic totals with climate records."""
    district = models.ForeignKey(ZambianDistrict, on_delete=models.CASCADE)
    date = models.DateField()
    epi_week = models.IntegerField(validators=[MinValueValidator(1)])
    reporting_year = models.IntegerField()
    
    # Clinical Metrics from NMEC records
    rdt_confirmations = models.IntegerField(default=0)
    
    # Meteorological Metrics from Open-Meteo API data stream
    rainfall_mm = models.FloatField(default=0.0)
    avg_temperature_c = models.FloatField(default=25.0)

    class Meta:
        unique_together = ('district', 'date') # Composite structural safety constraint

    def __str__(self):
        return f"{self.district.name} - Week {self.epi_week} ({self.reporting_year})"
