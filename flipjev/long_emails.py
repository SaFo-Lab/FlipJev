"""Twenty multi-paragraph office emails, the targets of the email searches on long emails.

Each email has five or six paragraphs of ordinary prose, and the decisive sentence sits in a middle paragraph, so a
suffix is a small part of the text. Ten emails mention a machine learning course (label yes) and ten do not (label
no); the negatives are hard: a golf course, a machine learning product or job, a course on another subject, or a
machine learning mention and an unrelated course in separate sentences.
"""

import re

# Phrases that count as a mention of a machine learning course; the last one is the Chinese "machine learning course".
MENTION = [
    r"machine[- ]learning (course|class|seminar|module)",
    r"(course|class|seminar|module) (in|on|about) machine[- ]learning",
    r"\bML (course|class)\b",
    r"machine learning \((course|elective)\)",
    r"introduction to machine learning",
    r"\u673a\u5668\u5b66\u4e60(\u8bfe\u7a0b|\u8bfe)",
]

# Each email: subject, paragraphs (the decisive sentence sits in a middle one),
# index of that paragraph, stratum and the reason for the label.
POSITIVE = [
    ("Facilities update for the autumn term", [
        "Hello everyone. This note collects the building changes that take effect after the mid-term break, so please read it before you plan your week. The porters have asked me to send one message rather than several, and I will post the same text on the noticeboard by the lifts.",
        "The east staircase will be closed from Monday to Thursday while the handrails are replaced. During those four days, use either the main entrance or the ramp behind the bicycle shed. The lift in the north wing keeps its usual service hours, but it will pause for an inspection on Wednesday between ten and eleven.",
        "Teaching rooms have moved around a little. Room 204 is now a quiet study space, and the seminars that used to meet there have been spread across the second floor. The machine learning course has been given room 118 for the rest of the term, which is the larger room with the sloping desks and the projector that actually works.",
        "Catering will run a reduced menu on Tuesday because the main oven is being serviced. Sandwiches, soup and the salad bar will be available as usual, and the coffee machine on the ground floor has been refilled. If your group has booked lunch for a visitor, please confirm the numbers with the kitchen by Monday noon.",
        "Finally, a reminder about the lost property cupboard by reception. It is emptied at the end of every month, and anything unclaimed goes to the charity shop on the high street. There are currently three umbrellas, a bicycle helmet and a rather good pair of gloves waiting for their owners.",
        "Thank you for your patience while the work is finished. If anything in this note affects a booking you have already made, send me a short reply and I will sort it out with the porters directly.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
    ("Budget lines to confirm before Friday", [
        "Hi Priya. Before I send the departmental return to the finance office, I need you to confirm a handful of lines. Nothing here should be a surprise, but the office rejects the whole form if a single figure is missing, so it is worth ten minutes now rather than an argument in November.",
        "The travel budget is the one I am least sure about. We have two conference trips booked, one in March and one in June, and the June flights were paid from last year's allocation because of the early booking discount. I have listed them under this year anyway, with a note, because that is what the finance office asked for last time.",
        "Teaching costs are straightforward apart from one item. The machine learning course needs a second set of licences for the lab machines, which is the four hundred pounds sitting on line fourteen. I have flagged it as recurring, since we will need the same licences next year unless the university changes supplier.",
        "Equipment is under budget by roughly six hundred pounds, mostly because the replacement printer came out of the central fund rather than ours. I suggest we leave the surplus where it is rather than moving it, because the finance office treats a transfer as a new request and that means another form.",
        "Hospitality has one unusual entry: the catering for the open day, which was larger than planned because more visitors came than we expected. The receipts are in the shared folder under October, scanned and legible, and I have written the visitor count on the cover sheet.",
        "If the figures look right to you, reply with a simple yes and I will submit on Thursday afternoon. If you want to change anything, please tell me the line number rather than the description, because several lines have similar names and I would rather not guess.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
    ("Handover notes while I am on leave", [
        "Morning all. I am away for the next three weeks and wanted to leave clear notes rather than a list of half-finished thoughts. Most of this is routine, but there are two or three items where a delay would cause real trouble, and I have marked those.",
        "First, the visitor from the partner institute arrives on the eighth. Her hotel is booked and paid, but nobody has yet agreed to meet her at the station. It is a twenty minute walk or a short taxi ride, and she has asked for the taxi because she is bringing equipment.",
        "Second, teaching cover. Two of my sessions fall inside the three weeks. Daniel has kindly agreed to take the reading group, and the ML class on Wednesday afternoons will be covered by Rosa, who taught it two years ago and still has the materials. Both of them have the room booking and the attendance list.",
        "Third, the shared drive is nearly full, which is why saving large files has become slow. I have asked the IT office to increase the quota, and the ticket number is in my signature file. If they ask for a justification, the honest answer is that the scanned archives from the summer are larger than anyone expected.",
        "Fourth, expenses. I have submitted everything up to the end of last week. There is one receipt I cannot find, for a train ticket in August, and I have written it off rather than hold up the rest of the claim.",
        "I will read email perhaps twice a week, but slowly and without any enthusiasm. For anything urgent, Daniel is the right person and knows where my notes are. Thank you, and see you in November.",
    ], 2, "abbreviation", "A middle paragraph names the ML class."),
    ("Room bookings for the week of the twelfth", [
        "Hello. The booking system is being migrated on the eleventh, so for one week I am managing rooms by hand from this spreadsheet. Please do not book anything through the old system after Friday, because those bookings will not be carried across and you will arrive to find someone else in the room.",
        "Monday is quiet. The only fixed booking is the staff meeting at nine in the small committee room, which has been moved from its usual slot because the chair is travelling back that morning and cannot make an earlier start.",
        "Tuesday is the difficult day. Three groups want the lecture theatre in the afternoon, and I have given it to the largest. The course on machine learning will run in the theatre from two until four, and the other two groups have the second floor seminar rooms, which between them seat about forty.",
        "Wednesday and Thursday are mostly free, apart from the external audit on Thursday morning. The auditors need a room with a lockable door and a table large enough to spread out paper files, so they have the room behind reception for the whole morning.",
        "Friday afternoon I have deliberately left empty. Every migration I have been involved in has produced at least one surprise on the first day, and it is easier to have space in the timetable than to move a class at short notice.",
        "If your booking is not in the list I circulated, it has not been carried over and you should tell me today. I would rather hear from ten people who are already in the list than from one person who is not.",
    ], 2, "paraphrase", "A middle paragraph names a course on machine learning."),
    ("Library reading lists: deadline and format", [
        "Dear colleagues. Reading lists for next term are due with the library by the end of the month. The library asks for them earlier every year, and the reason is genuinely practical: if a book has to be ordered from abroad it can take six weeks to arrive.",
        "The format has changed slightly. Each list now needs the edition and, where relevant, the chapter range, because last year several students bought a different edition and found the page numbers useless. The library has a template in the shared folder, and it is worth using rather than inventing your own.",
        "Some lists are already in. The machine learning course has submitted its list, which is short and sensible: two textbooks, both available electronically, and a handful of articles the library already licenses. If you want a model to copy, that one is a good example of a list students will actually read.",
        "For anything not held by the library, there is a small acquisitions budget. It is not large enough for every request, so the librarians prioritise items on required reading over items on further reading. If a book is genuinely essential, say so in the notes field rather than assuming it is obvious.",
        "There is also the question of scanning. The library can scan one chapter or five per cent of a book, whichever is greater, and no more. Requests above that limit are refused, politely but firmly, and the refusal usually arrives too late to arrange an alternative.",
        "Please send your list as a single attachment rather than in the body of an email. The librarians copy them into a database, and an attachment survives that process with its formatting intact, which the body of an email does not.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
    ("Open day: schedule and volunteers", [
        "Hi everyone. The open day is on the twenty-second and I now have enough volunteers to cover the day, so thank you to those who replied quickly. This note sets out the schedule and who is where, and I will bring printed copies on the morning.",
        "Visitors arrive from half past nine and are given a map, a timetable and a bag which is heavier than it needs to be. The welcome talk is at ten in the lecture theatre, and it finishes promptly at half past because the tours start immediately afterwards.",
        "Between eleven and one there are taster sessions in four rooms. The machine learning class is running a taster in room 118, with a demonstration that has been tested twice and still occasionally fails in an interesting way. The other three rooms have the usual mix of short talks and hands-on activities.",
        "Lunch is in the atrium from one until two. There is a vegetarian option, a vegan option and, I am told, something involving pulled jackfruit. Please eat with the visitors rather than retreating to the staff room, because the informal conversations over lunch are what people remember.",
        "The afternoon is tours and one-to-one conversations. Two of us should stay near the entrance throughout, because a surprising number of visitors arrive in the afternoon having missed the morning entirely, and they need someone to tell them what is still happening.",
        "We finish at four, and the tidying takes about half an hour if everyone helps and rather longer if they do not. There will be leftover cake, which is the only incentive I have available.",
    ], 2, "abbreviation", "A middle paragraph names the machine learning class."),
    ("IT maintenance window and what it affects", [
        "Good afternoon. The annual maintenance window is on Saturday from six in the morning until two in the afternoon. Most of you will notice nothing, because most of you sensibly do not work on Saturday mornings, but a few services are affected and one of them matters for teaching.",
        "The file server is being moved to new hardware. During the move the shared drives are read-only, so you can open files but not save them. If you are working on something that day, copy it to your own machine first and copy it back afterwards, rather than discovering the problem at the moment you press save.",
        "The virtual machine pool used by the machine learning course will be rebuilt during the same window. Anything saved inside those machines rather than on the shared drive will be lost, and the students have been told twice, which experience suggests means about half of them know.",
        "Email and the website are unaffected. The printing service will restart at some point during the morning, which means any job queued on Friday evening may print twice or not at all. If a document is confidential, do not leave it queued over the weekend.",
        "The wireless network will drop briefly at around eleven when the controllers are updated. Devices normally reconnect by themselves. Devices that do not normally reconnect by themselves are usually older printers, and the fix is to turn them off and on again.",
        "If something is still broken on Monday morning, please raise a ticket rather than sending me a message, because tickets are what the IT office actually looks at. I will chase anything that is still open by Tuesday.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
    ("Programme review: what the panel asked for", [
        "Dear all. The review panel met last week and their written feedback arrived this morning. The overall tone is positive, and I want to say clearly that the panel praised the teaching, which is a direct result of work many of you did without much recognition.",
        "They did raise three things. The first is assessment load in the second year, which they think is too heavy in the spring. This is not news to us, and the plan to move one piece of coursework into the autumn should resolve it, provided the timetable allows.",
        "The second is progression. The panel noted that students arrive at the machine learning module with very uneven mathematical preparation, and asked whether a bridging session before the module starts would help. I think it would, and I would rather we designed it than had one imposed.",
        "The third is feedback turnaround, where we are slower than the university target on two modules. The panel accepted that the marking load is uneven across the year but asked for a plan rather than an explanation, which is fair.",
        "There is also a small administrative point about how we record attendance, where our practice and the written procedure have drifted apart. The simplest fix is to change the written procedure to match what everyone actually does, and I will propose that at the next meeting.",
        "I will circulate a draft response before the end of the month. If you disagree with anything above, tell me early, because once the response is submitted it becomes the thing we are held to for three years.",
    ], 2, "explicit", "A middle paragraph names the machine learning module."),
    ("Timetable clashes reported by students", [
        "Hello. Students have reported five timetable clashes for next term, and I have looked at each one. Two are genuine, two are misunderstandings and one is a mistake in the published timetable that I have already corrected.",
        "The first genuine clash affects students taking both the statistics option and the design workshop, which now overlap by half an hour on Thursdays. The workshop can start half an hour later without difficulty, and the tutor has agreed, so that one is solved.",
        "The second genuine clash is harder. Six students are registered for both the machine learning course and the fieldwork block, and those two cannot move: the course needs the lab and the fieldwork needs the minibus, and neither is available at another time. I have asked the six students which they would prefer to move to next year.",
        "The two misunderstandings are both about the reading week. The timetable shows sessions during reading week for one module, which is correct: that module runs a compressed schedule and always has. I have added a note to the published version so the next person to ask can read it rather than email.",
        "The published mistake was a room, not a time. One seminar was listed in a room that no longer exists as a teaching space, because it became an office in the summer. It now shows the correct room, and the students affected have been told.",
        "If you hear of another clash, please send me the student numbers rather than the names. The timetable system searches by number, and searching by name produces four people called Zhang and a great deal of wasted time.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
    ("Equipment loans over the vacation", [
        "Morning. A number of people have asked about borrowing equipment over the vacation, so here is the policy in one place. The short version is that most things can be borrowed, the paperwork is minimal, and the deadline for requests is the last Friday of term.",
        "Laptops and projectors are the most requested items and the easiest to lend. There are eleven laptops in the store, all with the standard build, and they go out on a first-come basis. If you need particular software installed, ask now rather than on the day, because installation needs a technician and technicians take leave too.",
        "Lab equipment is different. The measurement kit and the two workstations reserved for the machine learning course cannot leave the building, because the insurance covers them on site only. If you need to use them over the vacation, the building is open on weekdays and you can book a slot.",
        "Cameras and audio recorders can be borrowed for up to four weeks, which covers most fieldwork. Please return them charged and with the memory cards cleared, because the next borrower will otherwise spend an hour working out which files are theirs.",
        "Anything borrowed must be signed for, and the form asks where the item will be kept. This is not suspicion: it is what the insurer requires, and a form with the location left blank will be returned to you rather than processed.",
        "Requests go to the store rather than to me. I have copied Marek, who runs the store and who will be much faster than I would be at telling you whether a particular item is free.",
    ], 2, "explicit", "A middle paragraph names the machine learning course."),
]

NEGATIVE = [
    ("Away day: venue and travel", [
        "Hi all. The away day is confirmed for the fourteenth, and this note has the venue, the travel arrangements and the rough shape of the day. Please read the travel section even if you think you know the way, because the usual entrance is closed for building work.",
        "The venue is the conference centre at Brackenfield, about forty minutes out of town. There is a shuttle from the station at quarter past nine and quarter to ten, and it waits for the train rather than leaving on the dot, which is a small mercy.",
        "The centre sits next to a golf course, and the machine learning software they use to schedule the sprinklers apparently caused some excitement last summer when it watered the eighteenth green during a tournament. The staff tell this story to every visiting group, so you will hear it too.",
        "The morning is two sessions with a break, and the afternoon is a single long session followed by an hour that is officially for reflection and unofficially for going outside. Bring something waterproof, because the reflection happens outdoors whatever the weather.",
        "Lunch is included and dietary requirements were collected last month. If yours has changed, tell me by Monday and I will pass it on. The centre is genuinely good about this, unlike the place we used two years ago.",
        "Finally, the day finishes at four so that people with school pickups can get back in time. The shuttle returns at four and again at half past, and there is a bus from the village if you miss both.",
    ], 2, "overlap_golf", "A golf course and machine learning software, no class."),
    ("Recruitment: two posts to advertise", [
        "Dear colleagues. We have approval for two posts and I would like to advertise both before the end of the month. This note is mainly so that everyone knows what is being advertised and can suggest where to publicise it.",
        "The first post is a technician, half time, focused on the teaching labs. The person we appoint will spend most of their week preparing equipment and the rest of it fixing what the previous week broke. It is a genuinely important post and has been vacant for too long.",
        "The second post is a machine learning engineer to work on the data platform with the central team. It is a two year contract with the possibility of extension, and the work is mostly pipelines and monitoring rather than research, which we should be honest about in the advertisement.",
        "For both posts the closing date will be four weeks after publication, and interviews in the second week of December. I have asked for a panel of three, with one member from outside the department, which is the minimum the university now requires.",
        "If you know a good place to advertise, tell me. The standard channels reach the usual people, and the last two appointments both came through a personal recommendation rather than through the official listing.",
        "Please also think about who would be a good panel member. It is two half days and reading perhaps twenty applications, which I mention because people say yes and then discover the reading.",
    ], 2, "ml_without_course", "A machine learning engineer post, no class."),
    ("Evening classes: what staff can join", [
        "Hello everyone. The staff development office has published the evening classes for next term, and a few of them have places left. This is a short summary rather than the full brochure, which is on the intranet and which nobody reads.",
        "There is a pottery course on Mondays, which is popular and usually full by the second week. Two people from this floor took it last year and produced a number of bowls that are, charitably, unique. The course includes materials and the use of the kiln.",
        "On Tuesdays there is conversational Spanish at two levels, and on Thursdays a photography workshop that spends three weeks on cameras and the rest on editing. Both run for ten weeks and both count towards the annual development allowance.",
        "Separately, the data team has started a machine learning project on room usage, and they are looking for people willing to be interviewed about how they actually book rooms as opposed to how the policy says they should. It takes twenty minutes and they buy you a coffee.",
        "Places are allocated by the development office rather than by us, and they ask for a line of justification which nobody has ever been known to reject. Apply through the usual form before the end of the month.",
        "If a class you want is full, ask to be put on the waiting list rather than giving up. Attendance at evening classes drops after the first fortnight and places open up more often than you would think.",
    ], 3, "separate_mentions", "A pottery course and a machine learning project, separately."),
    ("Compliance training due this quarter", [
        "Dear all. The compliance deadline is the end of the quarter and the dashboard shows that rather a lot of us have not started. This note explains what is required, how long it takes and what happens if it is not done, which is the part people usually want to know.",
        "There are three modules. Data protection is the longest at about forty minutes, information security is twenty, and the AML course is fifteen minutes and applies only to staff who handle payments, which in this department is four people.",
        "The modules are on the training platform, and the platform works properly in a normal browser and badly on a phone. Progress is saved at the end of each section rather than continuously, so if you stop in the middle of a section you will repeat it.",
        "Our machine learning tools are inside the scope of the data protection module, which is why the examples this year involve automated decisions rather than filing cabinets. If you work with those tools, the examples are worth reading rather than clicking past.",
        "What happens if it is not done: the dashboard turns red, the compliance office emails your line manager, and after a further two weeks it emails the head of department. Nobody is dismissed for a late module, but the emails are tedious for everyone involved.",
        "The deadline is a Friday, and the platform is always slow on the last afternoon because several thousand people have the same idea. Doing it now takes the same amount of time and involves less waiting.",
    ], 1, "overlap_aml", "An AML course is anti-money laundering, not machine learning."),
    ("Conference report and what we should do next", [
        "Hi both. I am back from the conference and this is the report I promised, written on the train and therefore slightly disjointed. The short version is that it was worth attending and that two of the sessions have direct consequences for our work.",
        "The keynote was about evaluation, and the argument was that most published comparisons are not reproducible because the preprocessing is never described fully. This is uncomfortable and largely correct, and it affects how we should write up the current project.",
        "The most useful session was a workshop on deployment, where three groups described what went wrong after their models went live. All three had the same problem: the data changed and nobody noticed for months. We have the same weakness, and the fix is monitoring rather than a better model.",
        "I also finally met the group from Aarhus, who are doing something close to our work but with a different dataset. They are open to sharing, and I have their contact details. A joint paper is possible but probably premature.",
        "One practical thing: the evening event was at a restaurant where the main course arrived an hour late, which is how I ended up having a long and useful conversation with someone I had been trying to email for a year.",
        "I have put my notes and the slides I could find in the shared folder. The notes from the second afternoon are thin because the room was too hot to concentrate in.",
    ], 4, "overlap_meal", "A main course at dinner, not a class."),
    ("Boat trip and the river survey", [
        "Morning all. The river survey is on the ninth and I have finally sorted out the boat, which took longer than the survey will. This note has the timings, what to bring and the safety briefing, which is compulsory and mercifully short.",
        "We meet at the landing stage at eight. The boat holds nine and there are eight of us, so there is room for the equipment but not for anyone who turns up hoping to join. Please be punctual, because the lock keeper opens on the hour and the next opening is at nine.",
        "The survey itself is straightforward: samples at six points along the watercourse, photographs at each, and the flow readings which the sensor now records automatically. Last year we did the readings by hand and the machine learning model we trained on that data still has a suspicious gap where someone forgot.",
        "Bring waterproofs, sunscreen and lunch. The waterproofs are not optional even if the forecast is good, because the spray from the weir soaks everyone on the way back and there is nowhere to shelter.",
        "The safety briefing is at the landing stage before we board. It covers the lifejackets, what to do if someone goes in, and the fact that the water is colder than it looks in September. It takes five minutes.",
        "If you cannot come, tell me today rather than on the morning. Eight people is the right number for the work, and with six it becomes a long day for everyone.",
    ], 2, "overlap_watercourse", "A watercourse and a machine learning model, no class."),
    ("Medical practice: appointment changes", [
        "Dear patient. We are writing to tell you about changes to how appointments are booked at the practice from the first of next month. Please read this letter even if you rarely contact us, because the telephone number for cancellations has changed.",
        "Routine appointments can now be booked up to four weeks in advance, rather than two. This should reduce the number of people who cannot get an appointment at all, though it does mean the first week of each month is busy.",
        "Urgent appointments are unchanged: telephone at eight in the morning and you will be seen the same day. The triage nurse decides what counts as urgent, and the machine learning tool we use to sort incoming requests is an aid to that decision and never replaces it.",
        "Repeat prescriptions should be ordered through the app or the pharmacy rather than by telephone. If you have been prescribed a course of antibiotics, that is not a repeat prescription and should not be ordered this way.",
        "The practice will be closed for staff training on the afternoon of the eleventh. The out of hours service covers that afternoon and the number is on the back of this letter and on the door of the surgery.",
        "If you no longer wish to receive letters and would prefer email, please tell reception. We are trying to reduce printing, though we will keep writing to patients who prefer paper.",
    ], 2, "overlap_treatment", "A course of antibiotics and a machine learning tool, no class."),
    ("Summer school: draft programme", [
        "Hello. Here is the draft programme for the summer school. It is a draft in the real sense, meaning the shape is right and half the details are wrong, so please read it with that in mind and tell me what is missing.",
        "Week one is language and orientation. The mornings are a Spanish course for the visiting students, at two levels, and the afternoons are a mixture of campus tours, library inductions and the administrative business of registering forty people who all need a card.",
        "Week two is the project week. Students work in groups of four on a topic proposed by a member of staff, and present on the Friday. We have five topics and need two more, so if you have something that a small group could make progress on in four days, propose it.",
        "The social programme is thinner than last year because the budget is smaller. There is the welcome dinner, one trip to the coast and a farewell event which will be in the atrium rather than at a venue. Students have never complained about this, in my experience.",
        "One of the proposed project topics uses machine learning to sort the photograph archive, and the member of staff proposing it has asked for a room with decent machines. I have provisionally given them the lab on the Tuesday and Wednesday.",
        "Please send comments by the end of next week. After that the programme goes to the visiting institutions, and changing it then involves three sets of approvals and a certain amount of apologising.",
    ], 4, "separate_mentions", "A Spanish course and a machine learning project topic, separately."),
    ("Warehouse move: what happens to the archive", [
        "Dear colleagues. The warehouse lease ends in March and everything stored there has to move. This note explains what is happening to the paper archive, because several people have asked and the answer is more complicated than I expected.",
        "About two thirds of the boxes can be destroyed under the retention schedule. The schedule is dull but clear: most administrative paper goes after six years, and anything relating to a student record is kept far longer and moves to the central store.",
        "The remaining third has to be digitised, and this is where the delay is. The scanning contractor quoted for a straightforward job, then discovered that a great many pages are stapled, folded or both. Stapled pages cannot be fed through their machines and have to be handled individually.",
        "We are using machine learning to sort the scanned pages into categories, which works well for typed documents and poorly for handwriting. Pages it cannot classify go into a review queue, and the queue is currently rather long.",
        "For anyone who needs a document from the archive between now and March: ask early. Retrieval from the warehouse takes three working days at the moment and will take longer once the boxes start moving.",
        "There will be a skip in the yard for two weeks in February. Please do not use it for anything other than the archive, because the last skip filled with old office chairs within a day and the archive paper had nowhere to go.",
    ], 3, "ml_without_course", "Machine learning used for sorting scans, no class."),
    ("Driving lessons and the minibus permit", [
        "Hi Tom. Following our conversation, here is what I found out about the minibus permit. The short answer is that you need the assessment but not the full course, which is better news than I expected when I started asking.",
        "The university requires a separate assessment for anyone driving the minibus, even with a full licence. It takes about ninety minutes with an assessor, mostly manoeuvring and reversing, and there is a short written part on vehicle checks.",
        "If you have held your licence for less than three years, you also need a defensive driving course, which is a full day at the training centre. You have had yours for eleven years, so this does not apply to you, though the form asks the question in a confusing way.",
        "The assessment is booked through the transport office and there is usually a two week wait. They have asked me to remind everyone that turning up without the physical licence means the assessment is cancelled and charged to the department.",
        "Unrelated, but while I was on the phone: the transport office is trialling a machine learning system to predict which vehicles will fail their next inspection. They were very keen to tell me about it and I was mainly interested in the permit.",
        "Let me know which week suits you and I will ask them to book it. Once the assessment is done the permit lasts three years, so it is worth doing before the fieldwork season rather than during it.",
    ], 4, "course_on_other_subject", "A defensive driving course and a machine learning system, no class."),
]


def long_email_cases():
    cases = []
    for label, entries in (("yes", POSITIVE), ("no", NEGATIVE)):
        for i, (subject, paragraphs, index, stratum, rationale) in enumerate(entries, 1):
            body = "\n\n".join(paragraphs)
            case_id = f"long_{label}_{i:02d}"
            cases.append({"case_id": case_id + "/clean", "base_id": case_id, "label": label, "subject": subject,
                          "body": body, "stratum": stratum, "rationale": rationale, "difficulty": "long_natural",
                          "mention_paragraph": index, "paragraphs": len(paragraphs), "kind": "clean",
                          "variant": "clean", "family": "clean", "target": None, "control_case_id": None})
    return cases


def check(cases):
    """The decisive sentence sits in a middle paragraph, and only yes emails mention the course."""
    for case in cases:
        paragraphs = case["body"].split("\n\n")
        assert len(paragraphs) >= 5, case["case_id"]
        assert 0 < case["mention_paragraph"] < len(paragraphs) - 1, case["case_id"]
        sentences = [s.strip() for s in re.findall(r"[^.]+\.", case["body"])]
        matches = [s for s in sentences if any(re.search(m, s, re.I) for m in MENTION)]
        if case["label"] == "yes":
            assert matches, f"{case['case_id']} has no mention"
            assert any(m in paragraphs[case["mention_paragraph"]] for m in matches), case["case_id"]
        else:
            assert not matches, f"{case['case_id']} unexpectedly mentions a course: {matches}"
