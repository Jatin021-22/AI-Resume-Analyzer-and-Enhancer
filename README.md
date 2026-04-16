# 🤖 AI Resume Analyzer & Enhancer

<div align="center">

🚀 **AI-powered SaaS platform to analyze, optimize, and enhance resumes for real-world hiring systems**

<br/>

<p><b>Built for modern job seekers — combining NLP, ATS optimization, and intelligent feedback systems.</b></p>

<br/>

![Python](https://img.shields.io/badge/Python-Backend-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-AI-yellow?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=for-the-badge\&logo=postgresql\&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

<br/>

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [Tech Stack](#-technology-stack) · [Setup](#-setup) · [API](#-api-endpoints) · [Admin Panel](#-admin-panel) · [Future Scope](#-future-scope)

</div>

---

# 🧠 Overview

**AI Resume Analyzer & Enhancer** is a **production-ready SaaS platform** designed to simulate real-world hiring systems.

It helps users:

* 📊 Analyze resume quality
* 🎯 Match resumes with job roles
* 🤖 Improve ATS (Applicant Tracking System) score
* 💡 Get actionable feedback
* 🧾 Enhance resume content using AI

> This system bridges the gap between **candidate resumes and recruiter expectations** using AI-driven insights.

---

# ✨ Features

## 🔍 Core AI Features

| Feature               | Description                                              |
| --------------------- | -------------------------------------------------------- |
| 🧠 Semantic Matching  | Matches resume with job descriptions using embeddings    |
| 📊 ATS Score          | Calculates resume strength based on keywords & structure |
| 🎯 Role Detection     | Identifies best-fit job roles                            |
| 🧩 Skill Clustering   | Groups skills using ML (KMeans)                          |
| 📄 Resume Parsing     | Extracts structured data from PDFs                       |
| 📞 Contact Extraction | Detects email, phone, and key details                    |
| 🧱 Section Detection  | Identifies resume sections intelligently                 |

---

## 🚀 Product Features

| Feature             | Description                                       |
| ------------------- | ------------------------------------------------- |
| 💬 Feedback System  | Users can submit feedback directly from UI        |
| 🛠 Admin Panel      | View users, feedback, analytics, and system stats |
| 📊 Analytics API    | Rating distribution, feedback insights            |
| 🔐 Scalable Backend | Built with FastAPI + PostgreSQL                   |
| ⚡ High Performance  | Async APIs with optimized processing              |

---

## 🧠 AI Enhancement (Upcoming / In Progress)

* ✍️ Resume Rewriting using AI
* 🎨 Template-based Resume Builder
* 📄 PDF Resume Export
* 🤖 Smart Suggestions Engine

---

# 🏗️ Architecture

```text
Frontend (HTML/JS)
        ↓
FastAPI Backend
        ↓
AI/NLP Engine (spaCy + Transformers)
        ↓
PostgreSQL Database
        ↓
Admin Dashboard & Analytics
```

---

# 🧠 Technology Stack

## Backend

* FastAPI
* Python

## AI / NLP

* spaCy (NER + NLP)
* sentence-transformers (semantic similarity)
* scikit-learn (KMeans clustering)

## Database

* PostgreSQL
* SQLAlchemy ORM
* Alembic (migrations)

## DevOps

* Docker (optional setup)

## Data Processing

* pdfplumber (PDF parsing)
* Regex + NLP hybrid extraction

---

# ⚙️ Setup

```bash
# Clone repo
git clone https://github.com/your-username/AI-Resume-Analyzer-and-Enhancer.git

cd resumeai

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

---

# 🔌 API Endpoints

## Resume Analysis

```
POST /api/analyze
```

## Feedback System

```
POST /api/feedback/submit
GET  /api/admin/feedback
```

## Admin APIs

```
GET /api/admin/stats
GET /api/admin/users
GET /api/admin/analyses
```

---

# 🛠 Admin Panel

Access:

```
/admin
```

### Features:

* 📊 View system analytics
* 💬 Manage user feedback
* 👥 Monitor users
* 📈 Track resume analyses
* ✅ Resolve feedback issues

---

# 📈 Product Vision

This project is evolving into a **full SaaS platform** for:

* Resume optimization
* Career insights
* AI-powered job preparation

---

# 🚀 Future Scope

* 🧾 Resume Builder with drag-drop UI
* 🤖 AI Resume Rewriter
* 📊 Dashboard analytics (charts & insights)
* 🌍 Multi-user SaaS deployment
* 💳 Subscription model

---

# 🤝 Contributing

Contributions are welcome!
Feel free to fork and improve the project.

---

# 📄 License

MIT License

---

# 💡 Author

**Jatin Prajapati**

---

<div align="center">

⭐ If you like this project, give it a star!

</div>
