let buttons = document.querySelectorAll("button")

buttons.forEach(button => {
    button.addEventListener("click", (e) => {
        let activeButtons = document.querySelectorAll(".active")
        activeButtons.forEach(button => {
            if (button.parentElement == e.target.parentElement) {
                button.classList.remove("active")
            }
        })
        e.target.classList.add("active")
        setVideoSrc()
    })
})

let setVideoSrc = () => {
    let activeButtons = document.querySelectorAll(".active")
    potentialIndex = Array.from(activeButtons[0].parentElement.children).indexOf(activeButtons[0]) + 1
    momentumIndex = Array.from(activeButtons[1].parentElement.children).indexOf(activeButtons[1]) + 1

    console.log(potentialIndex)
    console.log(momentumIndex)

    let videoSrcNum = potentialIndex * 3 - (3 - momentumIndex)

    let videoTag = document.getElementById("video")
    videoTag.src = "./wave_packet_" + videoSrcNum.toString() + ".mp4"
}

