lista = [((-12.08827880531362, -77.00333227104468), (-12.088853248105407, -77.00267109337076)), 
    ((-12.08827880531362, -77.00333227104468), (-12.090861261164562, -77.00302885358975)),
    ((-12.088853248105407, -77.00267109337076), (-12.08877357696727, -77.00213299581841)), 
    ((-12.088853248105407, -77.00267109337076), (-12.090735371121855, -77.00246022530374)),
    ((-12.08877357696727, -77.00213299581841), (-12.088658176863062, -77.00159655403915)), 
    ((-12.08877357696727, -77.00213299581841), (-12.090032483961112, -77.00195597003123)), 
    ((-12.08877357696727, -77.00213299581841), (-12.090672426078285, -77.00179503749744)),
    ((-12.08877357696727, -77.00213299581841), (-12.089969538752124, -77.00143025708753)), 
    ((-12.089969538752124, -77.00143025708753), (-12.089894791297144, -77.000913931875)),
    ((-12.089894791297144, -77.000913931875), (-12.089854974453058, -77.00034533921)),
    ((-12.088658176863062, -77.00159655403915), (-12.088605722253796, -77.00110302760223)), 
    ((-12.088658176863062, -77.00159655403915), (-12.089938066142073, -77.00141952825196)),
    ((-12.088605722253796, -77.00110302760223), (-12.088521794857524, -77.00053439931617)), 
    ((-12.088605722253796, -77.00110302760223), (-12.089914196050678, -77.00089840648195)),
    ((-12.088521794857524, -77.00053439931617), (-12.088500813004352, -76.99996040661236)), 
    ((-12.088521794857524, -77.00053439931617), (-12.089847432544513, -77.00033701048461)),
    ((-12.089847432544513, -77.00033701048461), (-12.090440885368597, -77.00024597331294)),
    ((-12.090861261164562, -77.00302885358975), (-12.090735371121855, -77.00246022530374)), 
    ((-12.090861261164562, -77.00302885358975), (-12.093429933871267, -77.00272007069809)),
    ((-12.090735371121855, -77.00246022530374), (-12.090672426078285, -77.00179503749744)), 
    ((-12.090735371121855, -77.00246022530374), (-12.093399998663703, -77.00206904134797)),
    ((-12.090672426078285, -77.00179503749744), (-12.090452039744, -77.00026659600145)),
    ((-12.090672426078285, -77.00179503749744), (-12.091312334282412, -77.001757905116)), 
    ((-12.091312334282412, -77.001757905116), (-12.092849143109406, -77.00152317139099)), 
    ((-12.092849143109406, -77.00152317139099), (-12.093399998663703, -77.00145749771961)),
    ((-12.090452039744, -77.00026659600145), (-12.091091980857376, -77.00017003648118)),
    ((-12.091081490031799, -77.00018076531677), (-12.09168995723523, -77.00012712113883)), 
    ((-12.09168995723523, -77.00012712113883), (-12.09205713505035, -77.00008420579648)),
    ((-12.09205713505035, -77.00008420579648), (-12.092560645425694, -77.00000910229863)), 
    ((-12.092560645425694, -77.00000910229863), (-12.093085183299484, -76.99987499185383)),
    ((-12.091312334282412, -77.001757905116), (-12.0912493504341, -77.00113559935043)), 
    ((-12.0912493504341, -77.00113559935043), (-12.091091980857376, -77.00017003648118)),
    ((-12.0912493504341, -77.00113559935043), (-12.09179487247623, -77.00109268400809)), 
    ((-12.09179487247623, -77.00109268400809), (-12.092172540930509, -77.00107122633693)),
    ((-12.092172540930509, -77.00107122633693), (-12.092781005651684, -77.0009424803099)),
    ((-12.09179487247623, -77.00109268400809), (-12.09168995723523, -77.00012712113883)),
    ((-12.092172540930509, -77.00107122633693), (-12.09205713505035, -77.00008420579648)),
    ((-12.092849143109406, -77.00152317139099), (-12.092781005651684, -77.0009424803099)), 
    ((-12.092781005651684, -77.0009424803099), (-12.092560645425694, -77.00000910229863)),
    ((-12.093429933871267, -77.00272007069809), (-12.093399998663703, -77.00206904134797)),
    ((-12.093399998663703, -77.00206904134797), (-12.093399998663703, -77.00145749771961)),
    ((-12.093399998663703, -77.00145749771961), (-12.093085183299484, -76.99987499185383)),
    ((-12.088521794857524, -77.00053439931617),(-12.088458678552852, -76.99995413516139 )),
    ((-12.090440885368597, -77.00024597331294),(-12.090380562754838, -76.99966032710414)),
    ((-12.093085183299484, -76.99987499185383),(-12.092966168683475, -76.99931586248526))]

lista2 = []

for item1, item2 in lista:
    if item1 not in lista2:
        lista.append(item1)
    if item2 not in lista2:
        lista.append(item2)

print(lista2)