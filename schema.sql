-- MariaDB dump 10.17  Distrib 10.4.12-MariaDB, for osx10.15 (x86_64)
--
-- Host: localhost    Database: MARIADB_CRM_SAAS
-- ------------------------------------------------------
-- Server version	10.4.12-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `actif_client`
--

DROP TABLE IF EXISTS `actif_client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `actif_client` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_client` int(11) NOT NULL,
  `id_type_actif` int(11) NOT NULL,
  `intitule` varchar(255) DEFAULT NULL,
  `valeur_initiale` decimal(15,2) DEFAULT NULL,
  `date_acquisition` date DEFAULT NULL,
  `valeur` decimal(15,2) DEFAULT NULL,
  `devise` varchar(10) DEFAULT NULL,
  `date_eval` date DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_actif_client_client` (`id_client`),
  KEY `fk_actif_client_type` (`id_type_actif`),
  CONSTRAINT `fk_actif_client_client` FOREIGN KEY (`id_client`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_actif_client_type` FOREIGN KEY (`id_type_actif`) REFERENCES `type_actif` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `avis`
--

DROP TABLE IF EXISTS `avis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `avis` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `reference` varchar(255) DEFAULT NULL,
  `date` datetime DEFAULT NULL,
  `id_affaire` int(11) DEFAULT NULL,
  `id_etape` int(11) DEFAULT NULL,
  `etat` int(11) DEFAULT NULL,
  `entree` decimal(15,2) DEFAULT NULL,
  `sortie` decimal(15,2) DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_avis_affaire` (`id_affaire`),
  KEY `fk_avis_etape` (`id_etape`),
  CONSTRAINT `fk_avis_affaire` FOREIGN KEY (`id_affaire`) REFERENCES `mariadb_affaires` (`id`),
  CONSTRAINT `fk_avis_etape` FOREIGN KEY (`id_etape`) REFERENCES `avis_regle` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=33547 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `avis_regle`
--

DROP TABLE IF EXISTS `avis_regle`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `avis_regle` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `etape` int(11) DEFAULT NULL,
  `nom` varchar(100) DEFAULT NULL,
  `editable` tinyint(4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `charge_client`
--

DROP TABLE IF EXISTS `charge_client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `charge_client` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_client` int(11) NOT NULL,
  `id_type_charge` int(11) NOT NULL,
  `intitule` varchar(255) DEFAULT NULL,
  `montant` decimal(15,2) DEFAULT NULL,
  `frequence` varchar(50) DEFAULT NULL,
  `date_debut` date DEFAULT NULL,
  `date_fin` date DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_charge_client_client` (`id_client`),
  KEY `fk_charge_client_type` (`id_type_charge`),
  CONSTRAINT `fk_charge_client_client` FOREIGN KEY (`id_client`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_charge_client_type` FOREIGN KEY (`id_type_charge`) REFERENCES `type_charge` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `dette_client`
--

DROP TABLE IF EXISTS `dette_client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dette_client` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_client` int(11) NOT NULL,
  `id_type_dette` int(11) NOT NULL,
  `intitule` varchar(255) DEFAULT NULL,
  `capital_initial` decimal(15,2) DEFAULT NULL,
  `date_souscription` date DEFAULT NULL,
  `capital_restant` decimal(15,2) DEFAULT NULL,
  `taux_interet` decimal(5,2) DEFAULT NULL,
  `mensualite` decimal(15,2) DEFAULT NULL,
  `date_debut` date DEFAULT NULL,
  `date_fin` date DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_dette_client_client` (`id_client`),
  KEY `fk_dette_client_type` (`id_type_dette`),
  CONSTRAINT `fk_dette_client_client` FOREIGN KEY (`id_client`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_dette_client_type` FOREIGN KEY (`id_type_dette`) REFERENCES `type_dette` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `documents`
--

DROP TABLE IF EXISTS `documents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `documents` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nom` varchar(255) DEFAULT NULL,
  `niveau` varchar(100) DEFAULT NULL,
  `obsolescence_annees` int(11) DEFAULT NULL,
  `risque` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `documents_client`
--

DROP TABLE IF EXISTS `documents_client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `documents_client` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_client` int(11) NOT NULL,
  `nom_client` varchar(255) DEFAULT NULL,
  `id_document_base` int(11) NOT NULL,
  `nom_document` varchar(255) DEFAULT NULL,
  `date_creation` timestamp NULL DEFAULT NULL,
  `date_obsolescence` timestamp NULL DEFAULT NULL,
  `obsolescence` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_documents_client_client` (`id_client`),
  KEY `fk_documents_client_document` (`id_document_base`),
  CONSTRAINT `fk_documents_client_client` FOREIGN KEY (`id_client`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_documents_client_document` FOREIGN KEY (`id_document_base`) REFERENCES `documents` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1586 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `documents_clients_obsoletes`
--

DROP TABLE IF EXISTS `documents_clients_obsoletes`;
/*!50001 DROP VIEW IF EXISTS `documents_clients_obsoletes`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE TABLE `documents_clients_obsoletes` (
  `id` tinyint NOT NULL,
  `id_client` tinyint NOT NULL,
  `nom_client` tinyint NOT NULL,
  `id_document_base` tinyint NOT NULL,
  `nom_document` tinyint NOT NULL,
  `date_creation` tinyint NOT NULL,
  `date_obsolescence` tinyint NOT NULL,
  `obsolescence` tinyint NOT NULL,
  `email_client` tinyint NOT NULL
) ENGINE=MyISAM */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `donnees_esg_etendu`
--

DROP TABLE IF EXISTS `donnees_esg_etendu`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `donnees_esg_etendu` (
  `isin` varchar(20) NOT NULL,
  `nom` varchar(255) DEFAULT NULL,
  `wasteEfficiency` decimal(10,4) DEFAULT NULL,
  `waterEfficiency` decimal(10,4) DEFAULT NULL,
  `executivePay` decimal(15,2) DEFAULT NULL,
  `boardIndependence` decimal(5,2) DEFAULT NULL,
  `environmentalGood` decimal(15,4) DEFAULT NULL,
  `socialGood` decimal(15,4) DEFAULT NULL,
  `environmentalHarm` decimal(15,4) DEFAULT NULL,
  `socialHarm` decimal(15,4) DEFAULT NULL,
  `numberOfEmployees` int(11) DEFAULT NULL,
  `avgPerEmployeeSpend` decimal(15,2) DEFAULT NULL,
  `pctFemaleBoard` decimal(5,2) DEFAULT NULL,
  `pctFemaleExec` decimal(5,2) DEFAULT NULL,
  `genderPayGap` decimal(5,2) DEFAULT NULL,
  `boardGenderDiversity` decimal(5,2) DEFAULT NULL,
  `ghgIntensityValue` decimal(15,4) DEFAULT NULL,
  `biodiversity` decimal(15,4) DEFAULT NULL,
  `emissionsToWater` decimal(15,4) DEFAULT NULL,
  `hazardousWaste` decimal(15,4) DEFAULT NULL,
  `scope1And2CarbonIntensity` decimal(15,4) DEFAULT NULL,
  `scope3CarbonIntensity` decimal(15,4) DEFAULT NULL,
  `carbonTrend` decimal(10,4) DEFAULT NULL,
  `temperatureScore` decimal(5,2) DEFAULT NULL,
  `exposureToFossilFuels` decimal(5,2) DEFAULT NULL,
  `renewableEnergy` decimal(5,2) DEFAULT NULL,
  `climateImpactRevenue` decimal(15,4) DEFAULT NULL,
  `climateChangePositive` decimal(15,4) DEFAULT NULL,
  `climateChangeNegative` decimal(15,4) DEFAULT NULL,
  `climateChangeNet` decimal(15,4) DEFAULT NULL,
  `naturalResourcePositive` decimal(15,4) DEFAULT NULL,
  `naturalResourceNegative` decimal(15,4) DEFAULT NULL,
  `naturalResourceNet` decimal(15,4) DEFAULT NULL,
  `pollutionPositive` decimal(15,4) DEFAULT NULL,
  `pollutionNegative` decimal(15,4) DEFAULT NULL,
  `pollutionNet` decimal(15,4) DEFAULT NULL,
  `avoidingWaterScarcity` decimal(15,4) DEFAULT NULL,
  `sfdrBiodiversityPAI` decimal(15,4) DEFAULT NULL,
  `controversialWeapons` tinyint(4) DEFAULT NULL,
  `violationsUNGC` tinyint(4) DEFAULT NULL,
  `processesUNGC` tinyint(4) DEFAULT NULL,
  `noteE` decimal(4,2) DEFAULT NULL,
  `noteS` decimal(4,2) DEFAULT NULL,
  `noteG` decimal(4,2) DEFAULT NULL,
  PRIMARY KEY (`isin`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_affaires`
--

DROP TABLE IF EXISTS `mariadb_affaires`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_affaires` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_personne` int(11) DEFAULT NULL,
  `ref` varchar(255) DEFAULT NULL,
  `date_debut` timestamp NULL DEFAULT NULL,
  `date_cle` timestamp NULL DEFAULT NULL,
  `SRRI` int(11) DEFAULT NULL,
  `frais_courtier` decimal(15,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_affaires_personne` (`id_personne`),
  CONSTRAINT `fk_affaires_personne` FOREIGN KEY (`id_personne`) REFERENCES `mariadb_clients` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2252 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_clients`
--

DROP TABLE IF EXISTS `mariadb_clients`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_clients` (
  `id` int(11) NOT NULL,
  `nom` varchar(255) DEFAULT NULL,
  `prenom` varchar(255) DEFAULT NULL,
  `SRRI` int(11) DEFAULT NULL,
  `telephone` varchar(50) DEFAULT NULL,
  `adresse_postale` text DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `email` (`email`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_evenement`
--

DROP TABLE IF EXISTS `mariadb_evenement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_evenement` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `type_id` int(11) NOT NULL,
  `client_id` int(11) DEFAULT NULL,
  `affaire_id` int(11) DEFAULT NULL,
  `support_id` int(11) DEFAULT NULL,
  `date_evenement` datetime NOT NULL,
  `statut` varchar(50) NOT NULL DEFAULT 'à faire',
  `commentaire` text DEFAULT NULL,
  `utilisateur_responsable` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_evenement_type` (`type_id`),
  KEY `fk_evenement_client` (`client_id`),
  KEY `fk_evenement_affaire` (`affaire_id`),
  KEY `fk_evenement_support` (`support_id`),
  CONSTRAINT `fk_evenement_affaire` FOREIGN KEY (`affaire_id`) REFERENCES `mariadb_affaires` (`id`),
  CONSTRAINT `fk_evenement_client` FOREIGN KEY (`client_id`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_evenement_support` FOREIGN KEY (`support_id`) REFERENCES `mariadb_support` (`id`),
  CONSTRAINT `fk_evenement_type` FOREIGN KEY (`type_id`) REFERENCES `mariadb_type_evenement` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_evenement_envoi`
--

DROP TABLE IF EXISTS `mariadb_evenement_envoi`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_evenement_envoi` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `evenement_id` int(11) NOT NULL,
  `canal` varchar(100) NOT NULL,
  `destinataire` varchar(255) NOT NULL,
  `objet` varchar(255) DEFAULT NULL,
  `contenu` text DEFAULT NULL,
  `date_envoi` datetime NOT NULL DEFAULT current_timestamp(),
  `statut` varchar(50) DEFAULT 'préparé',
  `modele_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_envoi_evenement` (`evenement_id`),
  KEY `fk_envoi_modele` (`modele_id`),
  CONSTRAINT `fk_envoi_evenement` FOREIGN KEY (`evenement_id`) REFERENCES `mariadb_evenement` (`id`),
  CONSTRAINT `fk_envoi_modele` FOREIGN KEY (`modele_id`) REFERENCES `mariadb_modele_document` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_evenement_intervenant`
--

DROP TABLE IF EXISTS `mariadb_evenement_intervenant`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_evenement_intervenant` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `evenement_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_intervenant_evenement` (`evenement_id`),
  CONSTRAINT `fk_intervenant_evenement` FOREIGN KEY (`evenement_id`) REFERENCES `mariadb_evenement` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_evenement_lien`
--

DROP TABLE IF EXISTS `mariadb_evenement_lien`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_evenement_lien` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `evenement_source_id` int(11) NOT NULL,
  `evenement_cible_id` int(11) NOT NULL,
  `type_lien` varchar(100) NOT NULL,
  `role` varchar(100) NOT NULL,
  `nom_intervenant` varchar(255) NOT NULL,
  `contact` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_lien_evenement_source` (`evenement_source_id`),
  KEY `fk_lien_evenement_cible` (`evenement_cible_id`),
  CONSTRAINT `fk_lien_evenement_cible` FOREIGN KEY (`evenement_cible_id`) REFERENCES `mariadb_evenement` (`id`),
  CONSTRAINT `fk_lien_evenement_source` FOREIGN KEY (`evenement_source_id`) REFERENCES `mariadb_evenement` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_evenement_statut`
--

DROP TABLE IF EXISTS `mariadb_evenement_statut`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_evenement_statut` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `evenement_id` int(11) NOT NULL,
  `statut_id` int(11) NOT NULL,
  `date_statut` datetime NOT NULL DEFAULT current_timestamp(),
  `utilisateur_responsable` varchar(255) DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_statut_evenement` (`evenement_id`),
  KEY `fk_statut_type` (`statut_id`),
  CONSTRAINT `fk_statut_evenement` FOREIGN KEY (`evenement_id`) REFERENCES `mariadb_evenement` (`id`),
  CONSTRAINT `fk_statut_type` FOREIGN KEY (`statut_id`) REFERENCES `mariadb_statut_evenement` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_historique_affaire_w`
--

DROP TABLE IF EXISTS `mariadb_historique_affaire_w`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_historique_affaire_w` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_affaire` int(11) NOT NULL,
  `date` timestamp NULL DEFAULT NULL,
  `valo` decimal(15,2) DEFAULT NULL,
  `mouvement` decimal(15,2) DEFAULT NULL,
  `sicav` decimal(15,2) DEFAULT NULL,
  `perf_sicav_hebdo` decimal(7,4) DEFAULT NULL,
  `perf_sicav_52` decimal(7,4) DEFAULT NULL,
  `volat` decimal(7,4) DEFAULT NULL,
  `annee` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_histo_affaire` (`id_affaire`),
  CONSTRAINT `fk_histo_affaire` FOREIGN KEY (`id_affaire`) REFERENCES `mariadb_affaires` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=240848 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_historique_personne_w`
--

DROP TABLE IF EXISTS `mariadb_historique_personne_w`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_historique_personne_w` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_personne` int(11) NOT NULL,
  `date` timestamp NULL DEFAULT NULL,
  `valo` decimal(15,2) DEFAULT NULL,
  `mouvement` decimal(15,2) DEFAULT NULL,
  `sicav` decimal(15,2) DEFAULT NULL,
  `perf_sicav_hebdo` decimal(7,4) DEFAULT NULL,
  `perf_sicav_52` decimal(7,4) DEFAULT NULL,
  `volat` decimal(7,4) DEFAULT NULL,
  `SRRI` int(11) DEFAULT NULL,
  `annee` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_histo_personne_client` (`id_personne`),
  CONSTRAINT `fk_histo_personne_client` FOREIGN KEY (`id_personne`) REFERENCES `mariadb_clients` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=93485 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_historique_support_w`
--

DROP TABLE IF EXISTS `mariadb_historique_support_w`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_historique_support_w` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `modif_quand` timestamp NULL DEFAULT NULL,
  `source` varchar(100) DEFAULT NULL,
  `id_source` int(11) DEFAULT NULL,
  `date` timestamp NULL DEFAULT NULL,
  `id_support` int(11) NOT NULL,
  `nbuc` decimal(18,6) DEFAULT NULL,
  `vl` decimal(15,6) DEFAULT NULL,
  `prmp` decimal(15,2) DEFAULT NULL,
  `valo` decimal(15,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_histo_support` (`id_support`),
  CONSTRAINT `fk_histo_support` FOREIGN KEY (`id_support`) REFERENCES `mariadb_support` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3264453 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_modele_document`
--

DROP TABLE IF EXISTS `mariadb_modele_document`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_modele_document` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nom` varchar(255) NOT NULL,
  `canal` varchar(50) NOT NULL,
  `objet` varchar(255) DEFAULT NULL,
  `contenu` text NOT NULL,
  `actif` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_statut_evenement`
--

DROP TABLE IF EXISTS `mariadb_statut_evenement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_statut_evenement` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_support`
--

DROP TABLE IF EXISTS `mariadb_support`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_support` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `code_isin` varchar(20) DEFAULT NULL,
  `nom` varchar(255) DEFAULT NULL,
  `cat_gene` varchar(100) DEFAULT NULL,
  `cat_principale` varchar(100) DEFAULT NULL,
  `cat_det` varchar(100) DEFAULT NULL,
  `cat_geo` varchar(100) DEFAULT NULL,
  `promoteur` varchar(255) DEFAULT NULL,
  `taux_retro` decimal(5,2) DEFAULT NULL,
  `SRRI` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `code_isin` (`code_isin`)
) ENGINE=InnoDB AUTO_INCREMENT=100044291 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mariadb_type_evenement`
--

DROP TABLE IF EXISTS `mariadb_type_evenement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mariadb_type_evenement` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(255) NOT NULL,
  `categorie` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=43 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mouvement`
--

DROP TABLE IF EXISTS `mouvement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mouvement` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `modif_quand` datetime DEFAULT NULL,
  `id_affaire` int(11) DEFAULT NULL,
  `id_mouvement_regle` int(11) DEFAULT NULL,
  `id_support` int(11) DEFAULT NULL,
  `id_avis` int(11) DEFAULT NULL,
  `montant_ope` decimal(15,2) DEFAULT NULL,
  `frais` decimal(15,2) DEFAULT NULL,
  `vl_date` datetime DEFAULT NULL,
  `date_sp` datetime DEFAULT NULL,
  `vl` decimal(15,6) DEFAULT NULL,
  `nb_uc` decimal(18,6) DEFAULT NULL,
  `etat` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_mouvement_affaire` (`id_affaire`),
  KEY `fk_mouvement_support` (`id_support`),
  KEY `fk_mouvement_regle` (`id_mouvement_regle`),
  CONSTRAINT `fk_mouvement_affaire` FOREIGN KEY (`id_affaire`) REFERENCES `mariadb_affaires` (`id`),
  CONSTRAINT `fk_mouvement_regle` FOREIGN KEY (`id_mouvement_regle`) REFERENCES `mouvement_regle` (`id`),
  CONSTRAINT `fk_mouvement_support` FOREIGN KEY (`id_support`) REFERENCES `mariadb_support` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=266297 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mouvement_regle`
--

DROP TABLE IF EXISTS `mouvement_regle`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mouvement_regle` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `code` varchar(10) DEFAULT NULL,
  `titre` varchar(255) DEFAULT NULL,
  `sens` int(11) DEFAULT NULL,
  `investi` tinyint(4) DEFAULT NULL,
  `prmp` tinyint(4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `revenu_client`
--

DROP TABLE IF EXISTS `revenu_client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `revenu_client` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_client` int(11) NOT NULL,
  `id_type_revenu` int(11) NOT NULL,
  `intitule` varchar(255) DEFAULT NULL,
  `montant` decimal(15,2) DEFAULT NULL,
  `frequence` varchar(50) DEFAULT NULL,
  `date_debut` date DEFAULT NULL,
  `date_fin` date DEFAULT NULL,
  `commentaire` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_revenu_client_client` (`id_client`),
  KEY `fk_revenu_client_type` (`id_type_revenu`),
  CONSTRAINT `fk_revenu_client_client` FOREIGN KEY (`id_client`) REFERENCES `mariadb_clients` (`id`),
  CONSTRAINT `fk_revenu_client_type` FOREIGN KEY (`id_type_revenu`) REFERENCES `type_revenu` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `situation_client`
--

DROP TABLE IF EXISTS `situation_client`;
/*!50001 DROP VIEW IF EXISTS `situation_client`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE TABLE `situation_client` (
  `id_client` tinyint NOT NULL,
  `nom` tinyint NOT NULL,
  `prenom` tinyint NOT NULL,
  `total_actifs` tinyint NOT NULL,
  `total_dettes` tinyint NOT NULL,
  `total_revenus` tinyint NOT NULL,
  `total_charges` tinyint NOT NULL,
  `solde_net` tinyint NOT NULL,
  `reste_a_vivre` tinyint NOT NULL
) ENGINE=MyISAM */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `type_actif`
--

DROP TABLE IF EXISTS `type_actif`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `type_actif` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `actif` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `type_charge`
--

DROP TABLE IF EXISTS `type_charge`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `type_charge` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `actif` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `type_dette`
--

DROP TABLE IF EXISTS `type_dette`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `type_dette` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `actif` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `type_revenu`
--

DROP TABLE IF EXISTS `type_revenu`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `type_revenu` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `libelle` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `actif` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping events for database 'MARIADB_CRM_SAAS'
--

--
-- Dumping routines for database 'MARIADB_CRM_SAAS'
--

--
-- Final view structure for view `documents_clients_obsoletes`
--

/*!50001 DROP TABLE IF EXISTS `documents_clients_obsoletes`*/;
/*!50001 DROP VIEW IF EXISTS `documents_clients_obsoletes`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `mariadb_crm_saas`.`documents_clients_obsoletes` AS select `dc`.`id` AS `id`,`dc`.`id_client` AS `id_client`,`dc`.`nom_client` AS `nom_client`,`dc`.`id_document_base` AS `id_document_base`,`dc`.`nom_document` AS `nom_document`,`dc`.`date_creation` AS `date_creation`,`dc`.`date_obsolescence` AS `date_obsolescence`,`dc`.`obsolescence` AS `obsolescence`,`c`.`email` AS `email_client` from (`mariadb_crm_saas`.`documents_client` `dc` join `mariadb_crm_saas`.`mariadb_clients` `c` on(`dc`.`id_client` = `c`.`id`)) where `dc`.`date_obsolescence` is not null and `dc`.`date_obsolescence` < current_timestamp() */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `situation_client`
--

/*!50001 DROP TABLE IF EXISTS `situation_client`*/;
/*!50001 DROP VIEW IF EXISTS `situation_client`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `mariadb_crm_saas`.`situation_client` AS select `c`.`id` AS `id_client`,`c`.`nom` AS `nom`,`c`.`prenom` AS `prenom`,coalesce(sum(distinct `ac`.`valeur`),0) AS `total_actifs`,coalesce(sum(distinct `dc`.`capital_restant`),0) AS `total_dettes`,coalesce(sum(distinct `rc`.`montant`),0) AS `total_revenus`,coalesce(sum(distinct `cc`.`montant`),0) AS `total_charges`,coalesce(sum(distinct `ac`.`valeur`),0) - coalesce(sum(distinct `dc`.`capital_restant`),0) AS `solde_net`,coalesce(sum(distinct `rc`.`montant`),0) - coalesce(sum(distinct `cc`.`montant`),0) AS `reste_a_vivre` from ((((`mariadb_crm_saas`.`mariadb_clients` `c` left join `mariadb_crm_saas`.`actif_client` `ac` on(`c`.`id` = `ac`.`id_client`)) left join `mariadb_crm_saas`.`dette_client` `dc` on(`c`.`id` = `dc`.`id_client`)) left join `mariadb_crm_saas`.`revenu_client` `rc` on(`c`.`id` = `rc`.`id_client`)) left join `mariadb_crm_saas`.`charge_client` `cc` on(`c`.`id` = `cc`.`id_client`)) group by `c`.`id`,`c`.`nom`,`c`.`prenom` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-09-26 14:41:16
