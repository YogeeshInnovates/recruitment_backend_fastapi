import re
import math
from collections import Counter
from typing import List, Dict, Tuple

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "if",
    "then", "else", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "just", "don", "now", "also", "about", "above", "after", "again",
    "against", "between", "into", "through", "during", "before", "below",
    "up", "down", "out", "off", "over", "under", "further", "once", "here",
    "there", "any", "being", "while", "because", "until", "although",
    "since", "unless", "yet", "still", "even", "already", "ever", "never",
    "always", "often", "sometimes", "usually", "however", "therefore",
    "moreover", "furthermore", "meanwhile", "nevertheless", "otherwise",
    "instead", "besides", "adding", "noted", "including", "well", "also"
}

SKILL_PATTERNS = [
    r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\btypescript\b',
    r'\bc\+\+\b', r'\bc#\b', r'\bruby\b', r'\bgo\b', r'\brust\b',
    r'\bswift\b', r'\bkotlin\b', r'\bscala\b', r'\bphp\b', r'\br\b',
    r'\bsql\b', r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b',
    r'\bredis\b', r'\belasticsearch\b', r'\boracle\b', r'\bsqlite\b',
    r'\bdynamodb\b', r'\bcassandra\b', r'\bneo4j\b',
    r'\breact\b', r'\bangular\b', r'\bvue\.?js\b', r'\bnext\.?js\b',
    r'\bnode\.?js\b', r'\bexpress\.?js\b', r'\bfastapi\b', r'\bdjango\b',
    r'\bflask\b', r'\bspring\b', r'\bdotnet\b', r'\blaravel\b',
    r'\brails\b', r'\bgraphql\b', r'\brest\b', r'\bapi\b',
    r'\baws\b', r'\bgcp\b', r'\bazure\b', r'\bdocker\b', r'\bkubernetes\b',
    r'\bk8s\b', r'\bterraform\b', r'\bansible\b', r'\bjenkins\b',
    r'\bcircleci\b', r'\bci/cd\b', r'\bgit\b', r'\bgithub\b',
    r'\bbitbucket\b', r'\bjenkins\b', r'\bdevops\b', r'\bcloud\b',
    r'\bmachine\s*learning\b', r'\bml\b', r'\bdeep\s*learning\b',
    r'\bartificial\s*intelligence\b', r'\bai\b', r'\bnlp\b',
    r'\btensorflow\b', r'\bpytorch\b', r'\bscikit-?learn\b',
    r'\bpandas\b', r'\bnumpy\b', r'\bspark\b', r'\bhadoop\b',
    r'\bdata\s*science\b', r'\bdata\s*analysis\b', r'\bdata\s*engineering\b',
    r'\betl\b', r'\bbig\s*data\b', r'\btableau\b', r'\bpower\s*bi\b',
    r'\bagile\b', r'\bscrum\b', r'\bjira\b', r'\btrello\b',
    r'\bhtml\b', r'\bcss\b', r'\bsass\b', r'\bbootstrap\b', r'\btailwind\b',
    r'\bwebpack\b', r'\bbabel\b', r'\bnpm\b', r'\byarn\b',
    r'\bphotoshop\b', r'\billustrator\b', r'\bfigma\b', r'\bsketch\b',
    r'\bux\b', r'\bui\b', r'\buser\s*experience\b', r'\buser\s*interface\b',
    r'\bcybersecurity\b', r'\bsecurity\b', r'\bpenetration\s*testing\b',
    r'\bfirewall\b', r'\bencryption\b', r'\bblockchain\b',
    r'\bproduct\s*management\b', r'\bproject\s*management\b',
    r'\bleadership\b', r'\bcommunication\b', r'\bteamwork\b',
    r'\bproblem.solving\b', r'\bcritical\s*thinking\b',
    r'\btime\s*management\b', r'\bpublic\s*speaking\b',
    r'\bexcel\b', r'\bword\b', r'\bpowerpoint\b', r'\boffice\b',
    r'\blinux\b', r'\bunix\b', r'\bwindows\b', r'\bmacos\b',
    r'\bandroid\b', r'\bios\b', r'\breact\s*native\b', r'\bflutter\b',
    r'\bmobile\s*development\b', r'\bweb\s*development\b',
    r'\bbackend\b', r'\bfrontend\b', r'\bfullstack\b', r'\bfull.stack\b',
    r'\bqa\b', r'\bquality\s*assurance\b', r'\btesting\b', r'\bautomation\b',
    r'\bselenium\b', r'\bjunit\b', r'\bpytest\b', r'\bcypress\b',
    r'\bsocket\.?io\b', r'\bwebsocket\b', r'\brabbitmq\b', r'\bkafka\b',
    r'\bnginx\b', r'\bapache\b', r'\biis\b',
    r'\blatex\b', r'\bmatlab\b', r'\bsas\b', r'\bstata\b',
]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\.\s\-\/]', ' ', text)
    tokens = text.split()
    return [t.strip('.-/') for t in tokens if len(t.strip('.-/+')) > 0]


def remove_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def extract_keywords(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    filtered = remove_stop_words(tokens)
    total = len(filtered) if filtered else 1
    counts = Counter(filtered)
    tf_scores = {}
    for word, count in counts.items():
        tf_scores[word] = count / total
    return tf_scores


def calculate_keyword_match(resume_keywords: Dict[str, float], jd_keywords: Dict[str, float]) -> float:
    if not jd_keywords:
        return 0.0

    jd_words = set(jd_keywords.keys())
    resume_words = set(resume_keywords.keys())

    if not jd_words:
        return 0.0

    matched = jd_words.intersection(resume_words)
    if not matched:
        return 0.0

    matched_weight = sum(jd_keywords[w] for w in matched)
    total_weight = sum(jd_keywords.values())

    if total_weight == 0:
        return 0.0

    raw_score = matched_weight / total_weight

    coverage = len(matched) / len(jd_words)
    score = (raw_score * 0.6 + coverage * 0.4) * 100
    return round(min(score, 100.0), 2)


def extract_skills_from_text(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for pattern in SKILL_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            skill = m.strip()
            if skill and skill not in found:
                found.append(skill)
    return found


def calculate_skills_overlap(resume_skills: List[str], jd_skills: List[str]) -> Tuple[List[str], List[str]]:
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)
    matched = list(resume_set.intersection(jd_set))
    missing = list(jd_set - resume_set)
    return matched, missing


def screen_resume(resume_text: str, job_description: str) -> Dict:
    jd_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    keyword_score = calculate_keyword_match(resume_keywords, jd_keywords)

    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(job_description)
    matched_skills, missing_skills = calculate_skills_overlap(resume_skills, jd_skills)

    skill_score = 0.0
    if jd_skills:
        skill_score = (len(matched_skills) / len(jd_skills)) * 100

    total_score = round(keyword_score * 0.5 + skill_score * 0.5, 2)

    if total_score >= 75:
        level = "strong"
    elif total_score >= 50:
        level = "moderate"
    elif total_score >= 25:
        level = "weak"
    else:
        level = "poor"

    analysis_parts = []
    analysis_parts.append(f"Keyword relevance: {keyword_score:.1f}%")
    analysis_parts.append(f"Skill match: {skill_score:.1f}% ({len(matched_skills)}/{len(jd_skills)} required skills found)")
    if matched_skills:
        analysis_parts.append(f"Matched: {', '.join(matched_skills[:10])}")
    if missing_skills:
        analysis_parts.append(f"Missing: {', '.join(missing_skills[:10])}")
    analysis = f"Overall match: {level}. " + ". ".join(analysis_parts)

    return {
        "score": total_score,
        "analysis": analysis,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def match_jobs(resume_text: str, jobs: List[Dict]) -> List[Dict]:
    resume_keywords = extract_keywords(resume_text)
    resume_skills = extract_skills_from_text(resume_text)

    results = []
    for job in jobs:
        jd_text = f"{job['title']} {job['description']} {job.get('required_skills', '')}"
        jd_keywords = extract_keywords(jd_text)
        jd_skills = extract_skills_from_text(jd_text)

        keyword_score = calculate_keyword_match(resume_keywords, jd_keywords)

        matched_skills, missing_skills = calculate_skills_overlap(resume_skills, jd_skills)
        skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0

        total_score = round(keyword_score * 0.5 + skill_score * 0.5, 2)

        reason_parts = []
        if matched_skills:
            reason_parts.append(f"Matching skills: {', '.join(matched_skills[:5])}")
        if missing_skills:
            reason_parts.append(f"Missing: {', '.join(missing_skills[:5])}")
        reason_parts.append(f"Keyword alignment: {keyword_score:.1f}%")
        reason = ". ".join(reason_parts) if reason_parts else "No significant match"

        results.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "score": total_score,
            "reason": reason,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def analyze_application(resume_text: str, job_description: str, cover_letter: str) -> Dict:
    resume_result = screen_resume(resume_text, job_description)
    resume_score = resume_result["score"]

    jd_keywords = extract_keywords(job_description)
    cl_keywords = extract_keywords(cover_letter)
    cover_letter_score = calculate_keyword_match(cl_keywords, jd_keywords)

    cl_skills = extract_skills_from_text(cover_letter)
    jd_skills = extract_skills_from_text(job_description)
    cl_matched, _ = calculate_skills_overlap(cl_skills, jd_skills)
    cl_skill_bonus = (len(cl_matched) / len(jd_skills) * 100) if jd_skills else 0

    cover_letter_score = round(cover_letter_score * 0.6 + cl_skill_bonus * 0.4, 2)

    overall_score = round(resume_score * 0.6 + cover_letter_score * 0.4, 2)

    if overall_score >= 75:
        recommendation = "STRONG_HIRE"
    elif overall_score >= 60:
        recommendation = "HIRE"
    elif overall_score >= 40:
        recommendation = "MAYBE"
    elif overall_score >= 25:
        recommendation = "CONSIDER_WITH_RESERVATION"
    else:
        recommendation = "NO_HIRE"

    strengths = []
    weaknesses = []

    if resume_result["matched_skills"]:
        strengths.append(f"Strong skill alignment: {', '.join(resume_result['matched_skills'][:5])}")
    if resume_score >= 60:
        strengths.append("Resume content aligns well with job requirements")
    if cover_letter_score >= 60:
        strengths.append("Cover letter addresses key job requirements")
    if not strengths:
        strengths.append("Demonstrates some relevant experience")

    if resume_result["missing_skills"]:
        weaknesses.append(f"Missing key skills: {', '.join(resume_result['missing_skills'][:5])}")
    if resume_score < 40:
        weaknesses.append("Resume has low keyword relevance to the position")
    if cover_letter_score < 30:
        weaknesses.append("Cover letter does not address key job requirements")
    if not weaknesses:
        weaknesses.append("Could provide more specific examples of relevant work")

    word_count = len(cover_letter.split())
    if word_count < 50:
        weaknesses.append("Cover letter is very brief")
    elif word_count > 800:
        weaknesses.append("Cover letter is excessively long")

    total_words = len(resume_text.split()) + len(cover_letter.split())
    summary = (
        f"Candidate shows {'strong' if overall_score >= 60 else 'moderate' if overall_score >= 40 else 'weak'} "
        f"alignment with the position (score: {overall_score}/100). "
        f"Resume match: {resume_score}/100. Cover letter match: {cover_letter_score}/100. "
        f"Recommendation: {recommendation}."
    )

    return {
        "overall_score": overall_score,
        "resume_score": resume_score,
        "cover_letter_score": cover_letter_score,
        "recommendation": recommendation,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "summary": summary,
    }
















