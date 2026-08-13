from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import SystemUser, ZambianDistrict, IntegratedMalariaData


@admin.register(SystemUser)
class SystemUserAdmin(UserAdmin):
    """Adds the custom role/district fields onto the stock Django user admin."""
    fieldsets = UserAdmin.fieldsets + (
        ('Malaria System Profile', {'fields': ('role', 'district_assignment')}),
    )
    list_display = ('username', 'role', 'district_assignment', 'is_staff')
    list_filter = ('role', 'is_staff', 'is_active')


@admin.register(ZambianDistrict)
class ZambianDistrictAdmin(admin.ModelAdmin):
    list_display = ('name', 'province', 'population', 'latitude', 'longitude')
    list_editable = ('population',)
    search_fields = ('name', 'province')
    list_filter = ('province',)


@admin.register(IntegratedMalariaData)
class IntegratedMalariaDataAdmin(admin.ModelAdmin):
    list_display = (
        'district', 'date', 'epi_week', 'reporting_year',
        'rdt_confirmations', 'rainfall_mm', 'avg_temperature_c',
    )
    list_filter = ('district', 'reporting_year')
    date_hierarchy = 'date'
    search_fields = ('district__name',)
