# CRM_SAAS — Maintenance rapide (nginx + uvicorn)

## 1) nginx (PDF en streaming)

Symptôme : téléchargement PDF qui se coupe (ex: 300–400 KB sur 500 KB), erreurs dans `/var/log/nginx/error.log` du type :
`open() "/var/lib/nginx/proxy/..." failed (13: Permission denied) while reading upstream`

Correctif : désactiver le proxy buffering pour le vhost crm_saas.

Template de référence :
- `/var/www/CRM_SAAS/config/nginx/crm_saas.template`

Pour restaurer rapidement :
```bash
sudo cp /var/www/CRM_SAAS/config/nginx/crm_saas.template /etc/nginx/sites-available/crm_saas
sudo nginx -t && sudo systemctl reload nginx
```

## 2) Relancer Uvicorn (port public 8100)

Script prêt à l'emploi :
- `/var/www/CRM_SAAS/scripts/restart_uvicorn_public.sh`

Usage :
```bash
/var/www/CRM_SAAS/scripts/restart_uvicorn_public.sh
```

Il stoppe l’ancien uvicorn puis démarre :
- host: `0.0.0.0`
- port: `8100`
- logs: `/var/www/CRM_SAAS/uvicorn.log`

## 3) Vérifications rapides

- Ports ouverts :
```bash
ss -ltnp | rg "8100|8101"
```

- Logs uvicorn :
```bash
tail -n 100 /var/www/CRM_SAAS/uvicorn.log
```

- Logs nginx erreurs :
```bash
tail -n 100 /var/log/nginx/error.log
```
