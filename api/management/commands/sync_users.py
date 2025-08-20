from django.core.management.base import BaseCommand
from api.models import Job, ApiUser
from django.utils import timezone

class Command(BaseCommand):
    help = 'Sync existing jobs with ApiUser records'
    
    def handle(self, *args, **options):
        jobs_without_users = Job.objects.filter(user__isnull=True)
        created_count = 0
        linked_count = 0
        
        for job in jobs_without_users:
            if job.ip_address:
                user, created = ApiUser.objects.get_or_create(
                    ip_address=job.ip_address,
                    defaults={
                        'first_seen': job.submission_time,
                        'last_seen': job.submission_time
                    }
                )
                if created:
                    created_count += 1
                
                job.user = user
                job.save(update_fields=['user'])
                linked_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Created {created_count} ApiUser records and linked {linked_count} jobs'
            )
        )