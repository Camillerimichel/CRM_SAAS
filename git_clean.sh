# 1. Créer une nouvelle branche clean
git checkout --orphan clean-main

# 2. Supprimer dump.sql et ajouter au .gitignore
rm -f dump.sql
echo "dump.sql" >> .gitignore
echo "*.sql" >> .gitignore

# 3. Ajouter tous les autres fichiers
git add .

# 4. Premier commit de la branche clean
git commit -m "Clean repository without SQL files"

# 5. Remplacer main par la branche clean
git branch -D main
git branch -m clean-main main

# 6. Push en force
git push origin main --force